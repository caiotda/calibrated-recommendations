import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

from calibratedRecs.constants import USER_COL, ITEM_COL, GENRE_COL
from calibratedRecs.calibrationUtils import calculate_genre_distribution, element_wise_sub_module, count_zero_in_both

from calibratedRecs.distributions import standardize_prob_distributions


def KL(p, q):
    return entropy(p, q)

def get_kl_divergence(
    dist_p: dict, dist_q: dict, epsilon: float = 1e-9
) -> float:
    """
    Calculates the KL divergence between two probability distributions.

    Parameters:
        dist_a (dict[str, float]): First probability distribution, mapping genre to probability.
        dist_b (dict[str, float]): Second probability distribution, mapping genre to probability.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        float: KL divergence value.
    """
    p_std, q_std = standardize_prob_distributions(dist_p, dist_q)
    p_values = np.array(list(p_std.values()))
    q_values = np.array(list(q_std.values()))
    p_clipped = np.clip(p_values, epsilon, None)
    q_clipped = np.clip(q_values, epsilon, None)
    p_normalized = p_clipped / p_clipped.sum()
    q_normalized = q_clipped / q_clipped.sum()
    return KL(p_normalized, q_normalized).item()

def CE_at_k(rec_list, user_history, item2genreMap, k=20):
    rec_at_k = rec_list[:k]
    # E se o tipo de calibração muda? A maneira de calcular a distribuição deveria mudar também.
    rec_dist = calculate_genre_distribution(rec_at_k, item2genreMap)
    user_history_dist = calculate_genre_distribution(user_history, item2genreMap)

    # by default, we consider every genre possible in the distribution. Because MACE is an average
    # of differences between p(g|u) and q(g|u), if we consider samples where genres -therefore p(g|u)
    # and q(g|u) = 0 - we would be underestimating the calibration error as the 0 differences would
    # Diminish the average calibration error observed.
    user_history_dist_filtered = {
        k: v for k, v in user_history_dist.items()
        if v > 0 or rec_dist.get(k, 0) > 0
    }
    rec_dist_filtered = {k:v for k, v in rec_dist.items() if v > 0 or user_history_dist.get(k, 0) > 0}

    
    # Checks the absolute difference for every genre pertaining to an item in the recommendation
    # list or in the user history
    distribution_shift = element_wise_sub_module(user_history_dist_filtered, rec_dist_filtered) 
    return np.mean(list(distribution_shift.values()))

def ace(rec_list, user_history, item2genreMap):
    N = len(rec_list)
    ACE = 0
    for k in range(1, N+1):
        ACE += CE_at_k(rec_list, user_history, item2genreMap, k)
    return ACE/N



def mace(df, user2history, recCol, item2genreMap, subset=None):

        # Select only the rows for the given subset of users, if provided
        if subset is not None:
            df = df[df[USER_COL].isin(subset)].reset_index(drop=True)

        num_users = len(df)
        ACE_U = 0
        for u in tqdm(df.index, total=num_users):
            row = df.iloc[u]
            u = row["user"]
            rec = row[recCol]
            history = user2history[u]
            ACE_U += ace(rec, history, item2genreMap)

        return ACE_U / num_users