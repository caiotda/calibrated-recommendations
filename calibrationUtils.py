import torch

from typing import Counter
from constants import USER_COL, ITEM_COL, GENRE_COL

UNKNOWN_GENRE = "(no genres listed)"

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Talvez faça mais sentido isso ficar dentro do construtor da classe calibration. Esse cara
# é totalmente estático.
def build_item_genre_distribution_tensor(df, distribution_mode="steck"):
    item2genre = df[[ITEM_COL, GENRE_COL]].drop_duplicates()
    all_genres = item2genre.explode("genres")["genres"].drop_duplicates().tolist()
    std_dict = {genre: 0 for genre in all_genres}
    item2genre["genre_dist"] = item2genre["genres"].apply(
        lambda genre_list: normalize_counter(Counter(genre_list))
    )
    genre_vector = (
        item2genre["genre_dist"]
        .apply(lambda count: merge_dicts(std_dict, count))
        .apply(lambda dictionary: dict(sorted(dictionary.items())))
        .apply(lambda dictionary: list(dictionary.values()))
    ).tolist()
    return torch.tensor(genre_vector, device=dev).double()


def build_tensors_from_df(df, weight_col):
    if weight_col not in df.columns:
        raise ValueError(f"Column '{weight_col}' not found in DataFrame.")
    w_u_i_steck_df = df[[USER_COL, ITEM_COL, weight_col]]
    user_weight_vector = list(w_u_i_steck_df.itertuples(index=None, name=None))
    pre_tensor = torch.tensor(user_weight_vector, device=dev, dtype=torch.long)
    x, y, weight = pre_tensor[:, 0].int(), pre_tensor[:, 1].int(), pre_tensor[:, 2]
    return x, y, weight


def build_weight_tensor(
    df,
    weight_col,
    n_users,
    n_items,
    user_tensor=None,
    item_tensor=None,
    ratings_tensor=None,
):
    if user_tensor is None or item_tensor is None or ratings_tensor is None:
        user_tensor, item_tensor, ratings_tensor = build_tensors_from_df(df, weight_col)

    w_u_i_tensor = torch.zeros(size=(n_users, n_items), dtype=torch.long, device=dev)
    w_u_i_tensor[user_tensor, item_tensor] = ratings_tensor
    w_u_i_tensor = w_u_i_tensor.double()

    return w_u_i_tensor


def build_user_genre_history_distribution(
    df, p_g_i, weight_col="rating", distribution_mode="steck"
):
    n_users = df[USER_COL].nunique()
    n_items = df[ITEM_COL].nunique()
    w_u_i_tensor = build_weight_tensor(df, weight_col, n_users=n_users, n_items=n_items)
    return (w_u_i_tensor @ p_g_i) / w_u_i_tensor.sum(dim=1, keepdim=True)


def update_candidate_list_genre_distribution(
    user, w_hat_u_i, item_distribution_tensor, candidate_list
):
    w_hat_norm = w_hat_u_i[user, candidate_list].reshape(-1, 1).sum(dim=1, keepdim=True)
    subset_q_g_u = (
        w_hat_u_i[user, candidate_list].reshape(-1, 1)
        * item_distribution_tensor[candidate_list]
        / w_hat_norm
    )
    return torch.nan_to_num(subset_q_g_u, 0)


def clip_tensors_at_k(user_tensor, rec_ids, rec_scores, k):
    rec_at_k = rec_ids[:, :k]
    rec_ids_index = rec_at_k.reshape(1, -1)

    scores_at_k = rec_scores[:, :k]
    scores_index = scores_at_k.reshape(1, -1)

    user_tensor_interleaved = user_tensor.repeat_interleave(k)

    return user_tensor_interleaved, rec_ids_index, scores_index


def get_weight(genres_list, df, col_name):
    return df.loc[genres_list.index[0], col_name]


def preprocess_genres(df, genre_col="genres"):
    new_df = df.copy()
    new_df[genre_col] = new_df[genre_col].map(
        lambda genre: None if genre == UNKNOWN_GENRE else genre.split("|")
    )
    return new_df


def normalize_counter(counter):
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()} if total > 0 else {}


def merge_dicts(dict1, dict2):
    return {
        key: dict1.get(key, 0) + dict2.get(key, 0) for key in set(dict1) | set(dict2)
    }


def element_wise_mult(dict1, dict2):
    return {
        key: dict1[key] * dict2[key]
        for key in dict1.keys() & dict2.keys()
        if dict1[key] != 0 and dict2[key] != 0
    }


def element_wise_sub_module(dict1, dict2):
    return {key: abs(dict1[key] - dict2[key]) for key in dict1.keys() & dict2.keys()}


def element_wise_mult_nonzero(a, b):
    return {key: a[key] * b[key] if a[key] != 0 else 0 for key in a}
