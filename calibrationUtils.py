from typing import Counter


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

def count_zero_in_both(dict1, dict2):
    """Return number of keys present in both dicts whose values are zero in both."""
    return sum(1 for k in set(dict1) & set(dict2) if dict1.get(k) == 0 and dict2.get(k) == 0)


def element_wise_sub_module(dict1, dict2):
    return {key: abs(dict1[key] - dict2[key]) for key in dict1.keys() & dict2.keys()}



def element_wise_mult_nonzero(a, b):
    return {key: a[key] * b[key] if a[key] != 0 else 0 for key in a}
