import torch

from typing import Counter
from constants import USER_COL, ITEM_COL, GENRE_COL

UNKNOWN_GENRE = "(no genres listed)"


# Talvez faça mais sentido isso ficar dentro do construtor da classe calibration. Esse cara
# é totalmente estático.
def build_item_genre_distribution_tensor(df):
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
    return torch.tensor(genre_vector).double()


def build_user_genre_history_distribution(df, p_g_i, weight_col="rating"):

    user_weight_df = df[[USER_COL, ITEM_COL, weight_col]]
    w_u_i_steck_df = user_weight_df[["user", "item", "rating"]]
    user_weight_vector = list(w_u_i_steck_df.itertuples(index=None, name=None))
    pre_tensor = torch.tensor(user_weight_vector)
    x, y, rating = pre_tensor[:, 0].int(), pre_tensor[:, 1].int(), pre_tensor[:, 2]
    n_users = x.max().item() + 1
    n_items = y.max().item() + 1

    w_u_i_tensor = torch.ones(size=(n_users, n_items), dtype=torch.long)
    w_u_i_tensor[x, y] = rating
    w_u_i_tensor = w_u_i_tensor.double()

    return (w_u_i_tensor @ p_g_i) / w_u_i_tensor.sum(dim=1, keepdim=True)


def build_weight_tensor(user_tensor, rec_ids, rec_scores, n_items, k):
    rec_at_k = rec_ids[:, :k]
    rec_ids_index = rec_at_k.reshape(1, -1)

    scores_at_k = rec_scores[:, :k]
    scores_index = scores_at_k.reshape(1, -1)

    user_tensor_interleaved = user_tensor.repeat_interleave(k)

    n_users = len(user_tensor)
    w_u_i = torch.zeros(size=(n_users, n_items))
    w_u_i[user_tensor_interleaved, rec_ids_index] = scores_index

    return w_u_i.to(torch.float64)


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
