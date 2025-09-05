from constants import ITEM_COL, USER_COL, GENRE_COL
from functools import reduce
from typing import Counter
import numpy as np
import pickle



UNKNOWN_GENRE = "(no genres listed)"


def get_weight(genres_list, df, col_name):
    return df.loc[genres_list.index[0], col_name]



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


def get_gleb_proportion(local_dist, global_dist):
    prod = element_wise_mult_nonzero(local_dist, global_dist)
    log_prod = {k: np.log(x) if x != 0 else x for k, x in prod.items()}
    e_c = {k: -1 * prod[k] * log_prod[k] for k in prod}

    return e_c



def get_gleb_distribution(ratings, weight_col='rating'):
    
    gleb_df = ratings[[USER_COL, ITEM_COL, GENRE_COL, weight_col]]
    user_genre_counter = gleb_df.groupby([USER_COL, ITEM_COL, weight_col]).agg(
        genres_count=(GENRE_COL, lambda genres_list: Counter((genre for genres in genres_list for genre in genres))),
    ).reset_index()

    user_history = user_genre_counter.groupby(USER_COL).agg({
        "genres_count": lambda dicts: reduce(merge_dicts, dicts)
    }).reset_index()
    user_history.loc[:, "prop_h(u)"] = user_history["genres_count"].apply(normalize_counter)



    gleb_df = user_genre_counter.merge(user_history[[USER_COL, "prop_h(u)"]], on=USER_COL)

    gleb_df.loc[:, "prop(g|i)"] = gleb_df["genres_count"].apply(normalize_counter)

    gleb_df["e_c_u_i"] = gleb_df.apply(lambda row: get_gleb_proportion(row["prop(g|i)"], row["prop_h(u)"]), axis=1)
    gleb_df["gleb_dist_tmp"] = [
        {k: row[weight_col] * v for k, v in row["e_c_u_i"].items()}
        for _, row in gleb_df.iterrows()
    ]

    gleb_df_per_user = gleb_df.groupby(USER_COL).agg({
        "gleb_dist_tmp": lambda dicts: reduce(merge_dicts, dicts),
    }).reset_index()

    gleb_df_per_user["e_c_u"] = gleb_df_per_user["gleb_dist_tmp"].apply(normalize_counter)

    gleb_df_per_user = gleb_df_per_user.rename(columns={"e_c_u": "p(g|u)"})

    return gleb_df_per_user[[USER_COL, "p(g|u)"]]




def create_prob_distribution_df(ratings, weight_col='w_c'):
    """
        This function recieves a ratings data frame (the only requirements are that it should contain
        userID, itemID, timestamp and genres columns), a weight function, which maps the importance of each
        item to the user (could be an operation on how recent was the item rated, the rating itself
        etc) and returns a dataframe mapping an userID to its genre preference distribution
    """
    df = ratings.copy()
    # Here we simply count the number of genres found per item and the weight w_u_i
    user_genre_counter = df.groupby([USER_COL, ITEM_COL, weight_col]).agg(
        genres_count=(GENRE_COL, lambda genres_list: Counter((genre for genres in genres_list for genre in genres))),
    ).reset_index().rename(columns={weight_col: 'w_u_i'})
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
    user_to_prob_distribution = user_genre_counter.groupby(USER_COL)['p(g|u)_tmp'].agg(lambda dicts: reduce(merge_dicts, dicts)).reset_index()


    normalization_per_user = user_genre_counter.groupby(USER_COL)['w_u_i'].sum()
    user_to_prob_distribution["w_u_i_sum"] = normalization_per_user

    # Finally, we normalize p(g|u)_tmp by \sum_{i \in H} w_u_i, obtaining Stecks calibration formulation
    user_to_prob_distribution["p(g|u)"] = user_to_prob_distribution.apply(lambda row: {k: v/row["w_u_i_sum"] for k, v in row["p(g|u)_tmp"].items()}, axis=1)

    return user_to_prob_distribution[[USER_COL, "p(g|u)"]]