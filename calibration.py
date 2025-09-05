import math
from functools import reduce
from collections import Counter
from typing import Any

import pandas as pd
from tqdm import tqdm
from torch import tensor, device, cuda

from constants import ITEM_COL, USER_COL, GENRE_COL
from metrics import standardize_prob_distributions, get_kl_divergence
from weight_functions import (
    get_linear_time_weight_rating,
    get_constant_weight,
    get_rating_weight,
    recommendation_twb_weighting,
    recommendation_score_weigthing,
)
from calibrationUtils import (
    normalize_counter,
    merge_dicts,
    get_gleb_distribution,
    create_prob_distribution_df,
    get_weight,
)

dev = device('cuda' if cuda.is_available() else 'cpu')


CALIBRATION_MODE_TO_COL_NAME = {
    "constant": "w_c",
    "rating": "w_rui",
    "linear_time": "w_twb"
}


CALIBRATION_MODE_TO_DATA_PREPROCESS_FUNCTION = {
    "constant": get_constant_weight,
    "rating": get_rating_weight,
    "linear_time": get_linear_time_weight_rating
}

CALIBRATION_MODE_TO_RECOMMENDATION_PREPROCESS_FUNCTION = {
    "constant": recommendation_score_weigthing,
    "rating": recommendation_score_weigthing,
    "linear_time": recommendation_twb_weighting
}


DISTRIBUTION_MODE_TO_FUCNTION = {
    'steck': create_prob_distribution_df,
    'gleb': get_gleb_distribution
}


class Calibration:
    def __init__(self, ratings_df, recommendation_df, weight='constant', distribution_mode='steck'):
            
        self._validate_modes(weight, distribution_mode)

        genre_importance_function = CALIBRATION_MODE_TO_DATA_PREPROCESS_FUNCTION[weight]


        self.weight_col_name = CALIBRATION_MODE_TO_COL_NAME[weight]
        self.distribution_function = DISTRIBUTION_MODE_TO_FUCNTION[distribution_mode]
        self.weight = weight
        ratings_df["constant"] = 1
        self.ratings_df = ratings_df.transform(genre_importance_function)
        self.user2history = self.ratings_df.groupby(USER_COL).agg({ITEM_COL: list}).to_dict()[ITEM_COL]
        self.item2genre_df = self._setup_data_preprocessing()
        self.candidates  = tensor(ratings_df.item.unique(), device=dev)

        # Transform item2genre_df to a dictionary mapping item to genres
        self.item2genreMap = dict(zip(self.item2genre_df[ITEM_COL], self.item2genre_df["genres"]))
        self.rec_df = self.preprocess_recommendation_for_calibration(recommendation_df)
        self.calibration_df = self._setup_calibration_df()
       
        self.item2genre_count_dict = self.item2genre_df.set_index(ITEM_COL)["genre_count"].to_dict()
        self.item_to_genre_dict_map =  dict(zip(self.item2genre_df[ITEM_COL], self.item2genre_df["genre_distribution"]))
        self.all_genres = set(self.item2genre_df.genres.explode().unique())
        self.default_genre_count = {genre: 0 for genre in self.all_genres}
        self.n_genres = len(self.all_genres)
        
    def _validate_modes(self, weight, distribution_mode):
        if weight not in CALIBRATION_MODE_TO_COL_NAME:
            raise ValueError(f"Invalid weight mode: {weight}. Must be one of {list(CALIBRATION_MODE_TO_COL_NAME.keys())}")
        if distribution_mode not in DISTRIBUTION_MODE_TO_FUCNTION:
            raise ValueError(f"Invalid distribution mode: {distribution_mode}. Must be one of {list(DISTRIBUTION_MODE_TO_FUCNTION.keys())}")


    def preprocess_recommendation_for_calibration(self, rec_df):
        df = rec_df.copy()
        df["rank"] = df.groupby(USER_COL).cumcount() + 1
        df[GENRE_COL] = df["top_k_rec_id"].astype(int).map(self.item2genreMap)
        weight_function = CALIBRATION_MODE_TO_RECOMMENDATION_PREPROCESS_FUNCTION[self.weight]
        df[f"rec_{self.weight_col_name}"] = df.apply(weight_function, axis=1)
        return df


    def _setup_calibration_df(self):
        history_genre_distribution =  self.distribution_function(self.ratings_df, self.weight_col_name)
        user_recommendations_genre_distribution = self.distribution_function(
            ratings=self.rec_df.rename(columns={"top_k_rec_id": ITEM_COL}),
            weight_col=f"rec_{self.weight_col_name}").rename(columns=({"p(g|u)": "q(g|u)"})
        )

        grouped = self.rec_df.groupby(USER_COL).agg({"top_k_rec_id": list, "top_k_rec_score": list})
        grouped["rec_id_2_score_map"] = grouped.apply(
                    lambda row: dict(zip(row["top_k_rec_id"], row["top_k_rec_score"])), axis=1
                )


        calibration_df = (
            history_genre_distribution
            .merge(user_recommendations_genre_distribution,"inner", USER_COL)
            .merge(grouped,"inner", USER_COL)
        )

        return calibration_df
    
    def _setup_data_preprocessing(self):
        df_copy = self.ratings_df.copy()
        df_copy["genres"] = df_copy["genres"].apply(tuple)
        item2genre = df_copy[["item", "genres"]].drop_duplicates()
        item2genre["genres"] = item2genre["genres"].apply(list)

        item2genre['genre_distribution'] = item2genre['genres'].apply(lambda genres: normalize_counter(Counter(genres)))
        item2genre['genre_count'] = item2genre['genres'].apply(lambda genres: dict(Counter(genres)))

        return item2genre
    
    def _update_candidate_list_genre_distribution(self, current_list_dist, new_item_dist):
        a, b =  standardize_prob_distributions(current_list_dist, new_item_dist)
        return  merge_dicts(a,b)
    
    def calculate_gender_distribution(self, item_list):
        genre_map = self.item2genreMap

        genre_counts = Counter([genre for item in item_list for genre in genre_map[item]])
        return {genre: count / sum(genre_counts.values()) for genre, count in genre_counts.items()}
    
    def CE_at_k(self, rec_list, user_history, k=20):
        rec_at_k = rec_list[:k]
        rec_dist = self.calculate_gender_distribution(rec_at_k)
        user_history_rec = self.calculate_gender_distribution(user_history)

        return get_kl_divergence(rec_dist, user_history_rec) / self.n_genres
    
    def ace(self, rec_list, user_history):
        N = len(rec_list)
        ACE = 0
        for k in range(1, N):
            ACE += self.CE_at_k(rec_list, user_history, k)
        return ACE/N

    
    def mace(self, is_calibrated=True, subset=None):

        # Select only the rows for the given subset of users, if provided
        df = self.calibration_df
        if subset is not None:
            df = df[df[USER_COL].isin(subset)].reset_index(drop=True)

        if(is_calibrated):
            col = "calibrated_rec"
        else:
            col = "top_k_rec_id"

        num_users = len(df)
        ACE_U = 0
        for u in tqdm(df.index, total=num_users):
            user = df.iloc[u]
            rec = user[col]
            history = self.user2history[u]
            ACE_U += self.ace(rec, history)

        return ACE_U / num_users

    
    def calibrate_for_users(self, subset=None):
        calibrated_rec = []
        calibrated_dist = []

        # Select only the rows for the given subset of users, if provided
        df = self.calibration_df
        if subset is not None:
            df = df[df[USER_COL].isin(subset)].reset_index(drop=True)

        for i in tqdm(df.index, total=len(df)):
            rec, dist = self.calibrate(df.iloc[i])

            calibrated_rec.append(rec)
            calibrated_dist.append(dist)

        df["calibrated_rec"] = calibrated_rec
        df["calibrated_dist"] = calibrated_dist

        if subset is not None:
            self.calibration_df.loc[df.index, "calibrated_rec"] = df["calibrated_rec"]
            self.calibration_df.loc[df.index, "calibrated_dist"] = df["calibrated_dist"]
        else:
            self.calibration_df = df

    #def calibration_error(self):



    def calibrate(self, row, k=20, _lambda = 0.99):
        history_dist = row["p(g|u)"]
        rec_to_relevancy_map = row["rec_id_2_score_map"]
        item_to_genre_count_map = self.item2genre_count_dict
        total_relevancy = 0.0
        calibrated_rec = []
        # Start out with a uniform distribution with P(x) = 0 for every gender
        # x
        candidate_distribution = self.default_genre_count

        # Gets recomendation ids
        candidates = list(rec_to_relevancy_map.keys())
        # Don´t stop until we have a candidate list of size k
        while(len(calibrated_rec) < k):
            objective = -math.inf
            best_candidate = None
            best_candidate_relevancy = 0

            current_candidate_list_genre_counter = candidate_distribution

            relevancy_so_far = total_relevancy
            # Greedily adds candidates to the calibrated list, always choosing the
            # item that maximizes the equation
            # I = (1-lambda)  * sum_relevance(list) + lambda * kl_div(history_dist, list)
            for candidate in candidates:
                # We first get the candidate item information: its predicted
                # relevancy and its genre distribution
                candidate_relevancy = rec_to_relevancy_map[candidate]

                candidate_genre_counter = item_to_genre_count_map[candidate]
                
                # Now we see the genre distribution if we consider candidate, alongside
                # the relevancy of the list if we consider it.
                updated_genre_counter_with_candidate = self._update_candidate_list_genre_distribution(current_candidate_list_genre_counter, candidate_genre_counter)
                relevancy_so_far_with_candidate = relevancy_so_far + candidate_relevancy

                # Turn the counter into a probability distribution to calculate the kl divergence
                # in reference to the users history
                updated_genre_distribution_with_candidate_normalized = normalize_counter(updated_genre_counter_with_candidate)
                kl_divergence_candidate = get_kl_divergence(history_dist, updated_genre_distribution_with_candidate_normalized)

                # Finally, we measure the Maximal Marginal Relevance
                MMR_of_candidate_list = (1-_lambda) * relevancy_so_far_with_candidate - _lambda  * kl_divergence_candidate

                if (MMR_of_candidate_list > objective):
                    best_candidate = candidate
                    best_candidate_relevancy = candidate_relevancy
                    objective = MMR_of_candidate_list
                    best_cand_distribution = updated_genre_counter_with_candidate
            # Commit to the found candidate


            # 1. Atualizar a lista calibrada com esse melhor candidato
            calibrated_rec.append(best_candidate)
            # 2. Atualizar a soma da relevancia da lista ate agora
            total_relevancy += best_candidate_relevancy
            # 3. Atualizar a distribuição de generos nessa nova lista
            candidate_distribution = best_cand_distribution
            # 4. Remover o candidato da lista
            candidates.remove(best_candidate)
        return calibrated_rec, normalize_counter(candidate_distribution)
        


    
    
        