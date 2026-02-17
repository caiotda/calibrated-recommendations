import math
import torch

from tqdm import tqdm
import numpy as np

from calibratedRecs.constants import ITEM_COL, USER_COL
from calibratedRecs.metrics import get_avg_kl_div, get_kl_divergence, mace

from calibratedRecs.mappings import validate_modes, DISTRIBUTION_MODE_TO_FUNCTION


from calibratedRecs.calibrationUtils import (
    build_item_genre_distribution_tensor,
    build_user_genre_history_distribution,
    build_weight_tensor,
    preprocess_dataframe_for_calibration,
)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Calibration:

    def __init__(
        self,
        ratings_df,
        recommendation_df,
        n_users,
        n_items,
        weight="constant",
        distribution_mode="steck",
        _lambda=0.99,
    ):
        validate_modes(weight, distribution_mode)
        self.weight = weight
        self._lambda = _lambda
        self.is_calibrated = False
        self.distribution_function = DISTRIBUTION_MODE_TO_FUNCTION[distribution_mode]
        self.ratings_df = preprocess_dataframe_for_calibration(ratings_df)
        self.recommendation_df = recommendation_df.rename(
            columns={"top_k_rec_id": ITEM_COL, "top_k_rec_score": "rating"}
        )
        self.calibration_df = self.recommendation_df.copy()
        exploded_recommendation_df = self.recommendation_df.explode(
            [ITEM_COL, "rating"]
        )

        self.n_users = n_users
        self.n_items = n_items
        self.weight_tensor_history = build_weight_tensor(
            self.ratings_df, weight_col=self.weight, n_users=n_users, n_items=n_items
        )
        self.default_dtype = self.weight_tensor_history.dtype
        self.weight_tensor_recommendation = build_weight_tensor(
            exploded_recommendation_df,
            weight_col="rating",
            n_users=n_users,
            n_items=n_items,
        )
        self.item_distribution_tensor = build_item_genre_distribution_tensor(
            self.ratings_df
        )

        self.user_history_tensor = build_user_genre_history_distribution(
            self.ratings_df,
            self.item_distribution_tensor,
            n_users=n_users,
            n_items=n_items,
            weight_col=self.weight,
        )
        self.rec_distribution_tensor = build_user_genre_history_distribution(
            self.calibration_df,
            self.item_distribution_tensor,
            n_users,
            n_items,
            weight_col="rating",
        )

    def get_avg_kl_div(self, source="calibrated"):
        realized_dist = (
            self.calibrated_rec_distribution_tensor
            if source == "calibrated"
            else self.rec_distribution_tensor
        )
        users = self.ratings_df[USER_COL].unique()
        return get_avg_kl_div(users, self.user_history_tensor, realized_dist)

    def _mace(self, k=1000):
        if self.is_calibrated:
            df = (
                self.calibration_df.groupby(USER_COL)
                .agg(lambda x: list(x)[:k])
                .reset_index()
            )
        else:
            df = (
                self.recommendation_df.groupby(USER_COL)
                .agg(lambda x: list(x)[:k])
                .reset_index()
            )
        return mace(
            df,
            p_g_u=self.user_history_tensor,
            p_g_i=self.item_distribution_tensor,
        )

    def calibrate_for_users(self):
        calibrated_rec_scores = []
        calibrated_rec = []

        df = (
            self.calibration_df.groupby(USER_COL)
            .agg({ITEM_COL: list, "rating": list})
            .reset_index()
        )
        # Filter out df to keep only users with history
        has_history = self.user_history_tensor.isnan().all(dim=1) == False
        users_with_history = torch.nonzero(has_history).squeeze().cpu().numpy()
        df = df[df[USER_COL].isin(users_with_history)].reset_index()
        rec_tensor = torch.tensor(df[ITEM_COL].tolist(), device=dev).int()
        score_tensor = torch.tensor(
            df["rating"].tolist(), dtype=torch.float32, device=dev
        )
        users_tensor = torch.tensor(df[USER_COL].tolist(), device=dev).int()

        for i in tqdm(df.index, total=len(df)):
            user = users_tensor[i].item()
            rec_score_list = score_tensor[i, :].tolist()
            recommendation_list = rec_tensor[i, :].tolist()
            (
                reranked_rec,
                calibrated_rec_score,
            ) = self.calibrate(user, recommendation_list, rec_score_list)
            # Zip, sort by score descending, then unzip
            calibrated_rec.append(list(reranked_rec))
            calibrated_rec_scores.append(list(calibrated_rec_score))
        df[ITEM_COL] = calibrated_rec
        df["rating"] = calibrated_rec_scores

        self.calibration_df = df.explode([ITEM_COL, "rating"])
        self.calibrated_rec_distribution_tensor = build_user_genre_history_distribution(
            self.calibration_df,
            self.item_distribution_tensor,
            self.n_users,
            self.n_items,
        )
        self.is_calibrated = True

    def calibrate(self, user, recommendation_list, rec_score_list, k=20):
        _lambda = self._lambda
        candidates = list(zip(recommendation_list, rec_score_list))
        assert (
            len(candidates) > 0
        ), "Candidate list is empty. Cannot calibrate an empty list."
        total_relevancy = 0.0
        calibrated_rec = []
        calibrated_rec_relevancies = []

        # Move tensors to CPU as calibration is inherently sequential
        item_dist_cpu = self.item_distribution_tensor.cpu().detach()
        weight_rec_cpu = self.weight_tensor_recommendation.cpu().detach()
        user_history = self.user_history_tensor[user].cpu()

        running_weighted_genre_sum = torch.zeros_like(item_dist_cpu[0])
        running_weight_total = 0.0

        # Gets recomendation ids
        # Don´t stop until we have a candidate list of size k
        while len(calibrated_rec) < k:
            objective = -math.inf
            best_candidate_relevancy = 0
            # Greedily adds candidates to the calibrated list, always choosing the
            # item that maximizes the equation
            # I = (1-lambda)  * sum_relevance(list) - lambda * kl_div(history_dist, list)
            for idx, (candidate, candidate_relevancy) in enumerate(candidates):

                # Optimized: We pull the individual weight and genre vector just once
                w_i = weight_rec_cpu[user, candidate]
                g_i = item_dist_cpu[candidate]

                # Calculates the genre distribution including the candidate.
                new_weight_total = running_weight_total + w_i
                if new_weight_total == 0:
                    continue  # Avoid division by zero; if no weights, candidate doesn't affect distribution
                candidate_list_distribution = (
                    running_weighted_genre_sum + (w_i * g_i)
                ) / new_weight_total

                # Gets KL divergence between user history genre distribution and candidate list
                kl_divergence_candidate = get_kl_divergence(
                    user_history, candidate_list_distribution
                )

                relevancy_so_far_with_candidate = total_relevancy + candidate_relevancy

                # Finally, we measure the Maximal Marginal Relevance
                MMR_of_candidate_list = (
                    (1 - _lambda) * relevancy_so_far_with_candidate
                    - _lambda * kl_divergence_candidate
                )

                if MMR_of_candidate_list > objective:
                    best_idx = idx
                    best_candidate = candidate
                    best_candidate_relevancy = candidate_relevancy
                    best_w_i = w_i
                    best_g_i = g_i
                    objective = MMR_of_candidate_list

            # Commit to the found candidate
            calibrated_rec.append(best_candidate)
            calibrated_rec_relevancies.append(best_candidate_relevancy)
            total_relevancy += best_candidate_relevancy

            # Update the running totals for the next slot's calculations
            running_weighted_genre_sum += best_w_i * best_g_i
            running_weight_total += best_w_i

            # Use pop with the index to avoid the value-collision bug
            candidates.pop(best_idx)

        return calibrated_rec, calibrated_rec_relevancies
