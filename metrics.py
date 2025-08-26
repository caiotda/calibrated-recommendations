from scipy.stats import entropy
import numpy as np

def KL(p, q):
    return entropy(p, q)

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
