import numpy as np


from functools import reduce
from typing import Counter
from calibratedRecs.constants import USER_COL, ITEM_COL, GENRE_COL
from calibratedRecs.calibrationUtils import element_wise_mult_nonzero, normalize_counter, merge_dicts

def get_gleb_proportion(local_dist, global_dist):
    prod = element_wise_mult_nonzero(local_dist, global_dist)
    log_prod = {k: np.log(x) if x != 0 else x for k, x in prod.items()}
    e_c = {k: -1 * prod[k] * log_prod[k] for k in prod}

    return e_c



# TODO: a princípio, essa função vai morrer. So preciso ver como encaixar a construção do peso
# e tensor de categorias de itens. Pelo que entendi, o p(g|i) muda um pouco se for gleb também.
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
