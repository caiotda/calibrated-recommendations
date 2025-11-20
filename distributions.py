import numpy as np


from functools import reduce
from typing import Counter
from calibratedRecs.constants import USER_COL, ITEM_COL, GENRE_COL
from calibratedRecs.calibrationUtils import element_wise_mult_nonzero, normalize_counter, merge_dicts

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



def standardize_prob_distributions(
    dist_a: dict, dist_b: dict
) -> tuple[dict, dict]:
    """
    Standardizes two probability distributions by ensuring they have the same keys,
    filling missing keys with zero, and sorting them.

    Parameters:
        dist_a (dict[str, float]): First probability distribution, mapping genre to probability.
        dist_b (dict[str, float]): Second probability distribution, mapping genre to probability.

    Returns:
        tuple[dict[str, float], dict[str, float]]: Tuple of standardized and sorted distributions,
        each mapping genre to probability.
    """
    all_keys = set(dist_a.keys()) | set(dist_b.keys())
    for key in all_keys:
        if key not in dist_a:
            dist_a[key] = 0
        if key not in dist_b:
            dist_b[key] = 0
    dist_a_sorted = dict(sorted(dist_a.items()))
    dist_b_sorted = dict(sorted(dist_b.items()))
    return dist_a_sorted, dist_b_sorted


def update_candidate_list_genre_distribution(current_list_dist, new_item_dist):
    a, b =  standardize_prob_distributions(current_list_dist, new_item_dist)
    return  merge_dicts(a,b)



DISTRIBUTION_MODE_TO_FUCNTION = {
    'steck': create_prob_distribution_df,
    'gleb': get_gleb_distribution
}
