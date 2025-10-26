import math
from collections import Counter

from tqdm import tqdm
from torch import tensor, device, cuda
from constants import ITEM_COL, USER_COL, GENRE_COL
from metrics import get_kl_divergence

from mappings import (
    CALIBRATION_MODE_TO_COL_NAME,
    CALIBRATION_MODE_TO_DATA_PREPROCESS_FUNCTION,
    CALIBRATION_MODE_TO_RECOMMENDATION_PREPROCESS_FUNCTION,
)

from calibrationUtils import normalize_counter

from distributions import update_candidate_list_genre_distribution, DISTRIBUTION_MODE_TO_FUCNTION

from metrics import mace

dev = device('cuda' if cuda.is_available() else 'cpu')

class Calibration:

    def _validate_modes(self, weight, distribution_mode):
        """
            Validates if the weight strategy and genre distribution mode
            have been implemented
        """
        if weight not in CALIBRATION_MODE_TO_COL_NAME:
            raise ValueError(f"Invalid weight mode: {weight}. Must be one of {list(CALIBRATION_MODE_TO_COL_NAME.keys())}")
        if distribution_mode not in DISTRIBUTION_MODE_TO_FUCNTION:
            raise ValueError(f"Invalid distribution mode: {distribution_mode}. Must be one of {list(DISTRIBUTION_MODE_TO_FUCNTION.keys())}")


    def set_calibration_modes(self):
        """
            If calibration mode (weight) and genre distribution are valid (distribution_mode),
            sets the corresponding genre importance function and distribution function.
        """
        self._validate_modes(self.weight, self.distribution_mode)
        self.genre_importance_function = CALIBRATION_MODE_TO_DATA_PREPROCESS_FUNCTION[self.weight]
        self.weight_col_name = CALIBRATION_MODE_TO_COL_NAME[self.weight]
        self.distribution_function = DISTRIBUTION_MODE_TO_FUCNTION[self.distribution_mode]

    def prepare_dataframe_for_calibration(self):
        """
            Applies the genre importance function, yielding 
        """
        self.ratings_df["constant"] = 1
        self.ratings_df[GENRE_COL] = self.ratings_df[GENRE_COL].apply(tuple)
        self.ratings_df = self.ratings_df.transform(self.genre_importance_function)
        self.candidates  = tensor(self.ratings_df.item.unique(), device=dev)

    def build_lookup_mappings(self):
        """
        Construct lookup maps used by calibration:
        - user2history: user -> list of items in their history
        - item2genre_count_dict: item -> genre count dict
        - default_genre_count: default zero-count dict for all genres
        - item2genreMap : item -> list of genres of the item
        """
        self.user2history = self.ratings_df.groupby(USER_COL).agg({ITEM_COL: list}).to_dict()[ITEM_COL]

        item2genre = self.ratings_df.copy()[[ITEM_COL, GENRE_COL]].drop_duplicates()
        item2genre[GENRE_COL] = item2genre[GENRE_COL].apply(list)

        item2genre['genre_distribution'] = item2genre[GENRE_COL].apply(lambda genres: normalize_counter(Counter(genres)))
        item2genre['genre_count'] = item2genre[GENRE_COL].apply(lambda genres: dict(Counter(genres)))

        self.item2genre_df = item2genre
        self.item2genreMap = dict(zip(self.item2genre_df[ITEM_COL], self.item2genre_df["genres"]))

        self.item2genre_count_dict = self.item2genre_df.set_index(ITEM_COL)["genre_count"].to_dict()
        all_genres = set(self.item2genre_df.genres.explode().unique())
        self.default_genre_count = {genre: 0 for genre in all_genres}

    def preprocess_recommendation_for_calibration(self):
        """
        Prepare recommendation DataFrame for calibration in-place.

        Adds per-user rank, maps item IDs to genres, and computes a per-row
        weight column used for calibration.

        """
        self.recommendation_df["rank"] = self.recommendation_df.groupby(USER_COL).cumcount() + 1
        self.recommendation_df[GENRE_COL] = self.recommendation_df["top_k_rec_id"].astype(int).map(self.item2genreMap)
        weight_function = CALIBRATION_MODE_TO_RECOMMENDATION_PREPROCESS_FUNCTION[self.weight]
        self.recommendation_df[f"rec_{self.weight_col_name}"] = self.recommendation_df.apply(weight_function, axis=1)

    def _setup_calibration_df(self):
        # Calculates p(g|u)
        history_genre_distribution =  self.distribution_function(self.ratings_df, self.weight_col_name)
        # Calculates q(g|u)
        user_recommendations_genre_distribution = self.distribution_function(
            ratings=self.recommendation_df.rename(columns={"top_k_rec_id": ITEM_COL}),
            weight_col=f"rec_{self.weight_col_name}").rename(columns=({"p(g|u)": "q(g|u)"})
        )
        # Sets up recommendation score per item - required to run MMR optimization
        grouped = self.recommendation_df.groupby(USER_COL).agg({"top_k_rec_id": list, "top_k_rec_score": list})
        grouped["rec_id_2_score_map"] = grouped.apply(
                    lambda row: dict(zip(row["top_k_rec_id"], row["top_k_rec_score"])), axis=1
                )


        # Output is a dataframe containing p(g|u), q(g|u) and the recommendation list, alongside each item
        # score.
        calibration_df = (
            history_genre_distribution
            .merge(user_recommendations_genre_distribution,"inner", USER_COL)
            .merge(grouped,"inner", USER_COL)
        )

        return calibration_df
    
    def __init__(self, ratings_df, recommendation_df, weight='constant', distribution_mode='steck', _lambda=0.99):
        self.ratings_df = ratings_df
        self.recommendation_df = recommendation_df
        self.weight = weight
        self.distribution_mode = distribution_mode
        self._lambda = _lambda

        self.set_calibration_modes()    
        self.prepare_dataframe_for_calibration()
        self.build_lookup_mappings()

        self.preprocess_recommendation_for_calibration()
        self.calibration_df = self._setup_calibration_df()
    def _mace(self, is_calibrated=True, subset=None):

        # Select only the rows for the given subset of users, if provided
        df = self.calibration_df
        user2history = self.user2history
        if(is_calibrated):
            col = "calibrated_rec"
        else:
            col = "top_k_rec_id"

        return mace(
            df,
            user2history,
            subset=subset,
            recCol=col,
            item2genreMap=self.item2genreMap,
        )
    
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

    def calibrate(self, row, k=20):
        _lambda = self._lambda
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
                updated_genre_counter_with_candidate = update_candidate_list_genre_distribution(current_candidate_list_genre_counter, candidate_genre_counter)
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
        


    
    
        