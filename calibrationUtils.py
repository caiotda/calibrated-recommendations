from constants import ITEM_COL, USER_COL, GENRE_COL
from typing import Counter
from weight_functions import (
    get_linear_time_weight_rating,
    get_constant_weight,
    get_rating_weight,
    recommendation_twb_weighting,
    recommendation_score_weigthing,
)



UNKNOWN_GENRE = "(no genres listed)"


def get_weight(genres_list, df, col_name):
    return df.loc[genres_list.index[0], col_name]



def calculate_genre_distribution(item_list, genre_map):

    genre_counts = Counter([genre for item in item_list for genre in genre_map[item]])
    return {genre: count / sum(genre_counts.values()) for genre, count in genre_counts.items()}


def preprocess_genres(df, genre_col="genres"):
    new_df = df.copy()
    new_df[genre_col] = new_df[genre_col].map(lambda genre: None if genre == UNKNOWN_GENRE else genre.split("|"))
    return new_df


def normalize_counter(counter):
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()} if total > 0 else {}


def merge_dicts(dict1, dict2):
    return {key: dict1.get(key, 0) + dict2.get(key, 0) for key in set(dict1) | set(dict2)}

def element_wise_mult(dict1, dict2):
    return {key: dict1[key] * dict2[key] for key in dict1.keys() & dict2.keys() if dict1[key] != 0 and dict2[key] != 0}



def element_wise_mult_nonzero(a, b):
    return {key: a[key] * b[key] if a[key] != 0 else 0 for key in a}




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