import numpy as np
from scipy.stats import entropy
import torch
<<<<<<< HEAD
=======

>>>>>>> 7311e45 (Vectorizes MACE)


from calibratedRecs.calibrationUtils import build_weight_tensor

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



def CE(weight_tensor, user_history_tensor, p_g_i):
    q_g_u_k = weight_tensor @ p_g_i / weight_tensor.sum(dim=1, keepdim=True)
    return torch.abs(user_history_tensor - q_g_u_k).mean(dim=1)

def mace(rec_df, n_items, p_g_u, p_g_i):
        
        rec_tensor = torch.tensor(rec_df["top_k_rec_id"].tolist()).long()
        score_tensor = torch.tensor(rec_df["top_k_rec_score"].tolist())
        user_tensor = torch.tensor(rec_df["user"].tolist()).long()

        N = rec_tensor.shape[1]
        sum_CEs_tensor = torch.zeros(size=(user_tensor.shape[0], ))
        for k in range(1, N+1):
            w_u_i_k = build_weight_tensor(user_tensor, rec_tensor, score_tensor, n_items, k)
            sum_CEs_tensor += CE(weight_tensor=w_u_i_k, user_history_tensor=p_g_u, p_g_i=p_g_i)
        ACE = sum_CEs_tensor / N
        return ACE.mean().item()