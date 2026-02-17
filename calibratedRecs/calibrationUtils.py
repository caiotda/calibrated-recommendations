import torch

from typing import Counter
from calibratedRecs.constants import USER_COL, ITEM_COL, GENRE_COL

from calibratedRecs.weight_functions import get_linear_time_weight_rating

UNKNOWN_GENRE = "(no genres listed)"

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_dataframe_for_calibration(df):
    processed_df = df.copy()
    processed_df[GENRE_COL] = processed_df[GENRE_COL].apply(tuple)
    processed_df["constant"] = 1
    # We shift scores to be strictly positive per user in order to
    # avoid zero denominators in the KL divergence calculation
    processed_df["rating"] = processed_df.groupby("user")["rating"].transform(
        lambda x: (x - x.min()) + 1e-8
    )
    # processed_df = get_linear_time_weight_rating(processed_df)
    return processed_df


def build_item_genre_distribution_tensor(df, distribution_mode="steck"):
    """
        Builds genre distribution tensor for each item (p_g_i, as in Steck's paper).
        Parameters
    ----------
        df : pandas.DataFrame
            DataFrame containing at least the columns named by the global constants ITEM_COL and GENRE_COL
        distribution_mode : str, optional
            Mode for calculating genre distribution.
            Currently only "steck" is supported, which creates a distribution based on the frequency of genres for each item. Default is "steck".
        Returns
        -------
        p_g_i : torch.FloatTensor
            Tensor of shape (n_items, n_genres) where each row represents the genre distribution
            for the corresponding item. The tensor is on the device referenced by the global variable `dev`.
    """
    item2genre = df[[ITEM_COL, GENRE_COL]].drop_duplicates()
    n_items = item2genre.item.max() + 1
    all_genres = item2genre.explode("genres")["genres"].drop_duplicates().tolist()
    n_genres = len(all_genres)
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
    dist_tensor = torch.tensor(genre_vector, device=dev, dtype=torch.float32)
    item_tensor = torch.tensor(item2genre.item.tolist(), device=dev, dtype=torch.long)
    p_g_i = torch.zeros(size=(n_items, n_genres), dtype=torch.float32, device=dev)
    p_g_i[item_tensor] = dist_tensor
    return p_g_i


def build_tensors_from_df(df, weight_col):
    """
    Build tensors for users, items and weights from a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least the columns named by the global constants USER_COL and ITEM_COL,
        and the column specified by weight_col.
    weight_col : str
        Name of the weight column that determines the importance of interaction (user, item)

    Returns
    -------
    x : torch.Tensor
        1-D integer tensor of user identifiers/indices with shape (N,).
    y : torch.Tensor
        1-D integer tensor of item identifiers/indices with shape (N,).
    weight : torch.Tensor
        1-D float32 tensor of weights with shape (N,) on the device referenced by the global variable `dev`.

    """
    if weight_col not in df.columns:
        raise ValueError(f"Column '{weight_col}' not found in DataFrame.")
    w_u_i_steck_df = df[[USER_COL, ITEM_COL, weight_col]]
    user_weight_vector = list(w_u_i_steck_df.itertuples(index=None, name=None))
    pre_tensor = torch.tensor(user_weight_vector, device=dev, dtype=torch.float32)
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
    """
    Create a dense user-item weight tensor.
    ----------
    Parameters

    df : pandas.DataFrame
        DataFrame used to construct tensors when user_tensor/item_tensor/ratings_tensor are None.
    weight_col : str
        Column name in df containing weight/rating values.
    n_users : int
        Number of users (first dimension of returned tensor).
    n_items : int
        Number of items (second dimension of returned tensor).
    Optionals:
    user_tensor : torch.LongTensor or array-like, optional
        1D indices of users for each interaction (overrides df-based construction).
    item_tensor : torch.LongTensor or array-like, optional
        1D indices of items for each interaction (overrides df-based construction).
    ratings_tensor : torch.Tensor or array-like, optional
        1D weights/ratings for each interaction (overrides df-based construction).

    Returns
    -------
    torch.FloatTensor of shape (n_users, n_items)
        Dense tensor  with weights populated at (user, item)
        locations; entries without interactions are zero.
    """
    if user_tensor is None or item_tensor is None or ratings_tensor is None:
        user_tensor, item_tensor, ratings_tensor = build_tensors_from_df(df, weight_col)
    w_u_i_tensor = torch.zeros(size=(n_users, n_items), dtype=torch.float32, device=dev)
    indices = (user_tensor, item_tensor)
    w_u_i_tensor.index_put_(indices, ratings_tensor, accumulate=True)

    return w_u_i_tensor


def build_user_genre_history_distribution(
    df,
    p_g_i,
    n_users,
    n_items,
    weight_col="rating",
    distribution_mode="steck",
):
    w_u_i_tensor = build_weight_tensor(df, weight_col, n_users=n_users, n_items=n_items)
    return (w_u_i_tensor @ p_g_i) / w_u_i_tensor.sum(dim=1, keepdim=True)


def clip_tensors_at_k(user_tensor, rec_ids, rec_scores, k):
    rec_at_k = rec_ids[:, :k]
    rec_ids_index = rec_at_k.reshape(1, -1)

    scores_at_k = rec_scores[:, :k]
    scores_index = scores_at_k.reshape(1, -1)

    user_tensor_interleaved = user_tensor.repeat_interleave(k)

    return user_tensor_interleaved, rec_ids_index, scores_index


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


def element_wise_mult_nonzero(a, b):
    return {key: a[key] * b[key] if a[key] != 0 else 0 for key in a}
