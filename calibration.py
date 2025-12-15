import math
import torch
from collections import Counter

from tqdm import tqdm
from constants import ITEM_COL, USER_COL, GENRE_COL
from metrics import get_kl_divergence

from mappings import validate_modes, DISTRIBUTION_MODE_TO_FUNCTION


from weight_functions import get_linear_time_weight_rating

from calibrationUtils import (
    build_item_genre_distribution_tensor,
    build_user_genre_history_distribution,
    build_weight_tensor,
    normalize_counter,
    update_candidate_list_genre_distribution,
)

from metrics import mace

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Calibration:

    # Posso reaproveitar essa funcao pro rec df tb, não?
    def preprocess_dataframe_for_calibration(self, df):
        processed_df = df.copy()
        processed_df[GENRE_COL] = processed_df[GENRE_COL].apply(tuple)
        processed_df["constant"] = 1
        processed_df = get_linear_time_weight_rating(processed_df)
        return processed_df

    def __init__(
        self,
        ratings_df,
        recommendation_df,
        weight="constant",
        distribution_mode="steck",
        _lambda=0.99,
    ):
        validate_modes(weight, distribution_mode)
        self.weight = weight
        self._lambda = _lambda
        self.distribution_function = DISTRIBUTION_MODE_TO_FUNCTION[distribution_mode]
        self.ratings_df = self.preprocess_dataframe_for_calibration(ratings_df)

        self.recommendation_df = recommendation_df.rename(
            columns={"top_k_rec_id": ITEM_COL, "top_k_rec_score": "rating"}
        )
        self.calibration_df = self.recommendation_df.copy()
        exploded_recommendation_df = self.recommendation_df.explode(
            [ITEM_COL, "rating"]
        )

        n_users = self.ratings_df[USER_COL].nunique()
        n_items = self.ratings_df[ITEM_COL].nunique()
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
            self.ratings_df, self.item_distribution_tensor
        )

    def _mace(self, is_calibrated=False):
        if is_calibrated:
            df = self.calibration_df
        else:
            df = self.recommendation_df
        return mace(
            df,
            p_g_u=self.user_history_tensor,
            p_g_i=self.item_distribution_tensor,
        )

    def calibrate_for_users(self):
        calibrated_rec_scores = []
        calibrated_rec = []

        # Select only the rows for the given subset of users, if provided
        df = self.calibration_df
        rec_tensor = torch.tensor(df[ITEM_COL].tolist(), device=dev).int()
        score_tensor = torch.tensor(df["rating"].tolist(), dtype=torch.float32, device=dev)
        users_tensor = torch.tensor(df[USER_COL].tolist(), device=dev).int()

        for i in tqdm(df.index, total=len(df)):
            user = users_tensor[i].item()
            rec_score_list = score_tensor[i, :].tolist()
            recommendation_list = rec_tensor[i, :].tolist()
            reranked_rec, calibrated_rec_score,  = self.calibrate(
                user, recommendation_list, rec_score_list
            )
            # Zip, sort by score descending, then unzip
            calibrated_rec.append(list(reranked_rec))
            calibrated_rec_scores.append(list(calibrated_rec_score))

        self.calibration_df[ITEM_COL] = calibrated_rec
        self.calibration_df["rating"] = calibrated_rec_scores


    def calibrate(self, user, recommendation_list, rec_score_list, k=20):
        _lambda = self._lambda
        # Trocar isso por tensor.
        candidates = list(zip(recommendation_list, rec_score_list))
        total_relevancy = 0.0
        calibrated_rec = []
        calibrated_rec_relevancies = []
        # Start out with a uniform distribution with P(x) = 0 for every gender
        # x

        user_history = self.user_history_tensor[user]
        # Gets recomendation ids
        # Don´t stop until we have a candidate list of size k
        while len(calibrated_rec) < k:
            objective = -math.inf
            best_candidate = None
            best_candidate_relevancy = 0
            relevancy_so_far = total_relevancy
            # Greedily adds candidates to the calibrated list, always choosing the
            # item that maximizes the equation
            # I = (1-lambda)  * sum_relevance(list) + lambda * kl_div(history_dist, list)
            for candidate, candidate_relevancy in candidates:

                # Now we see the genre distribution if we consider candidate, alongside
                # the relevancy of the list if we consider it.

                candidate_list_distribution = update_candidate_list_genre_distribution(
                    user,
                    self.weight_tensor_recommendation,
                    self.item_distribution_tensor,
                    calibrated_rec + [candidate],
                )

                # Turn the counter into a probability distribution to calculate the kl divergence
                # in reference to the users history
                kl_divergence_candidate = get_kl_divergence(
                    user_history, candidate_list_distribution
                )

                relevancy_so_far_with_candidate = relevancy_so_far + candidate_relevancy

                # Finally, we measure the Maximal Marginal Relevance
                MMR_of_candidate_list = (
                    (1 - _lambda) * relevancy_so_far_with_candidate
                    - _lambda * kl_divergence_candidate
                )
                if MMR_of_candidate_list > objective:
                    best_candidate = candidate
                    best_candidate_relevancy = candidate_relevancy
                    objective = MMR_of_candidate_list
            # Commit to the found candidate

            calibrated_rec.append(best_candidate)
            calibrated_rec_relevancies.append(best_candidate_relevancy)
            total_relevancy += best_candidate_relevancy
            candidates.remove((best_candidate, best_candidate_relevancy))

        return calibrated_rec, calibrated_rec_relevancies
