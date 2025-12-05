from typing import Counter
import torch

UNKNOWN_GENRE = "(no genres listed)"



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



def calculate_genre_distribution(item_list, genre_map):

    genre_counts = Counter([genre for item in item_list for genre in genre_map.get(item, UNKNOWN_GENRE)])
    return normalize_counter(genre_counts)


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


def element_wise_sub_module(dict1, dict2):
    return {key: abs(dict1[key] - dict2[key]) for key in dict1.keys() & dict2.keys()}



def element_wise_mult_nonzero(a, b):
    return {key: a[key] * b[key] if a[key] != 0 else 0 for key in a}
