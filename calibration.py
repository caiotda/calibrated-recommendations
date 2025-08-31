import math

from functools import reduce
from typing import Counter
from metrics import standardize_prob_distributions, get_kl_divergence

from tqdm import tqdm



from constants import ITEM_COL, USER_COL, GENRE_COL
from weight_functions import (
    get_linear_time_weight_rating,
    get_constant_weight,
    get_rating_weight,
    recommendation_twb_weighting,
    recommendation_score_weigthing,
)


import pandas as pd
from torch import tensor, device, cuda


dev = device('cuda' if cuda.is_available() else 'cpu')

def normalize_counter(counter):
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()} if total > 0 else {}


def merge_dicts(dict1, dict2):
    return {key: dict1.get(key, 0) + dict2.get(key, 0) for key in set(dict1) | set(dict2)}


def create_prob_distribution_df(ratings, weight_mode='w_c'):
    """
        This function recieves a ratings data frame (the only requirements are that it should contain
        userID, itemID, timestamp and genres columns), a weight function, which maps the importance of each
        item to the user (could be an operation on how recent was the item rated, the rating itself
        etc) and returns a dataframe mapping an userID to its genre preference distribution
    """
    df = ratings.copy()
    # Here we simply count the number of genres found per item and the weight w_u_i
    user_genre_counter = df.groupby([USER_COL, ITEM_COL]).agg(
        genres_count=(GENRE_COL, lambda genres_list: Counter((genre for genres in genres_list for genre in genres))),
        w_u_i=(GENRE_COL, lambda  genres_list: get_weight(genres_list, df, weight_mode))  
    )
    # We normalize the item count to obtain p(g|i)
    user_genre_counter["p(g|i)"] = user_genre_counter["genres_count"].apply(
        lambda genre_counts: {genre: count / sum(genre_counts.values()) for genre, count in genre_counts.items()}
    )

    # Here, we obtain w_u_i * p(g|i), basically obtaining the importance of a genre per user
    user_genre_counter["p(g|u)_tmp"] = user_genre_counter.apply(
        lambda row: {k: row["w_u_i"] * v for k, v in row["p(g|i)"].items()}, axis=1
    )

    # This step builds \sum_{i \in H} w_u_i * p(g|i), by combining the genres
    # found in the users history.
    user_to_prob_distribution = user_genre_counter.groupby(level=USER_COL)['p(g|u)_tmp'].agg(lambda dicts: reduce(merge_dicts, dicts)).reset_index()


    normalization_per_user = user_genre_counter.groupby(USER_COL)['w_u_i'].sum()
    user_to_prob_distribution["w_u_i_sum"] = normalization_per_user

    # Finally, we normalize p(g|u)_tmp by \sum_{i \in H} w_u_i, obtaining Stecks calibration formulation
    user_to_prob_distribution["p(g|u)"] = user_to_prob_distribution.apply(lambda row: {k: v/row["w_u_i_sum"] for k, v in row["p(g|u)_tmp"].items()}, axis=1)

    return user_to_prob_distribution[[USER_COL, "p(g|u)"]]

def get_weight(genres_list, df, col_name):
    return df.loc[genres_list.index[0], col_name]


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


class Calibration:
    def __init__(self, ratings_df, model, weight='constant'):
        assert weight in CALIBRATION_MODE_TO_DATA_PREPROCESS_FUNCTION.keys(), \
            f"weight must be one of {list(CALIBRATION_MODE_TO_DATA_PREPROCESS_FUNCTION.keys())}, got '{weight}'"
        self.weight_col_name = CALIBRATION_MODE_TO_COL_NAME[weight]
        self.weight = weight
        genre_importance_function = CALIBRATION_MODE_TO_DATA_PREPROCESS_FUNCTION[weight]
        ratings_df["constant"] = 1
        self.ratings_df = ratings_df.transform(genre_importance_function)
        self.user2history = self.ratings_df.groupby(USER_COL).agg({ITEM_COL: list}).to_dict()[ITEM_COL]
        self.weight_function = CALIBRATION_MODE_TO_COL_NAME[weight]
        self.item2genre_df = self._setup_data_preprocessing()
        self.candidates  = tensor(ratings_df.item.unique(), device=dev)

        # Transform item2genre_df to a dictionary mapping item to genres
        self.item2genreMap = dict(zip(self.item2genre_df[ITEM_COL], self.item2genre_df["genres"]))

        self.calibration_df = self._setup_calibration_df()
       
        self.item2genre_count_dict = self.item2genre_df.set_index(ITEM_COL)["genre_count"].to_dict()
        self.item_to_genre_dict_map =  dict(zip(self.item2genre_df[ITEM_COL], self.item2genre_df["genre_distribution"]))
        self.all_genres = set(self.item2genre_df.genres.explode().unique())
        self.default_genre_count = {genre: 0 for genre in self.all_genres}
        self.n_genres = len(self.all_genres)


    def _setup_calibration_df(self):
        history_genre_distribution =  create_prob_distribution_df(self.ratings_df, self.weight_function)
        history_genre_distribution[["top_k_rec_id", "top_k_rec_score"]] = history_genre_distribution.apply(
        lambda row: pd.Series(self.get_top_k_recommendations_for_user(row)), axis=1
        )

        user_history_genre_distribution_df_exploded = history_genre_distribution.explode(["top_k_rec_id", "top_k_rec_score"]).rename(columns={"top_k_rec_id": ITEM_COL})
        user_history_genre_distribution_df_exploded[GENRE_COL] = user_history_genre_distribution_df_exploded[ITEM_COL].astype(int).map(self.item2genreMap)

        user_recommendations_genre_distribution = create_prob_distribution_df(
            ratings=user_history_genre_distribution_df_exploded[[USER_COL, ITEM_COL, GENRE_COL, "top_k_rec_score"]],
            weight_mode="top_k_rec_score").rename(columns=({"p(g|u)": "q(g|u)"})
        )


        calibration_df = history_genre_distribution.merge(user_recommendations_genre_distribution,"inner", USER_COL)

        calibration_df["rec_id_2_score_map"] = calibration_df.apply(
            lambda row: dict(zip(row["top_k_rec_id"], row["top_k_rec_score"])), axis=1
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
    
    def get_top_k_recommendations_for_user(self, row):
        return self.model.predict(
            user=tensor(data=row["user"], device=dev),
            candidates=self.candidates,
            k=100
        )
    
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
        


    
    
        