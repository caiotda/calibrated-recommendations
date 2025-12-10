from calibrationUtils import clip_tensors_at_k
import numpy as np
from scipy.stats import entropy
import torch

from calibratedRecs.calibrationUtils import build_weight_tensor



def KL(p, q):
    return entropy(p, q)

def get_kl_divergence(dist_p: torch.Tensor, dist_q: torch.Tensor) -> float:
    """
    Calculates the KL divergence between two probability distributions represented as torch tensors.

    Parameters:
        dist_p (torch.Tensor): First probability distribution tensor.
        dist_q (torch.Tensor): Second probability distribution tensor.

    Returns:
        float: KL divergence value.
    """
    dist_p_array = dist_p.cpu()
    dist_q_array = dist_q.cpu()
    return KL(dist_p_array, dist_q_array).item()


def CE(weight_tensor, user_history_tensor, p_g_i):
    q_g_u_k = weight_tensor @ p_g_i / weight_tensor.sum(dim=1, keepdim=True)
    q_g_u_k_filled = torch.nan_to_num(q_g_u_k, nan=0.0)
    return torch.abs(user_history_tensor - q_g_u_k_filled).mean(dim=1)


def mace(rec_df, p_g_u, p_g_i):

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rec_tensor = torch.tensor(rec_df["top_k_rec_id"].tolist(), device=dev).int()
    score_tensor = torch.tensor(
        rec_df["top_k_rec_score"].tolist(), dtype=torch.long, device=dev
    )
    user_tensor = torch.tensor(rec_df["user"].tolist(), device=dev).int()

    n_users = p_g_u.shape[0]
    n_items = p_g_i.shape[0]

    N = rec_tensor.shape[1]
    sum_CEs_tensor = torch.zeros(size=(user_tensor.shape[0],), device=dev)
    for k in range(1, N + 1):
        user_tensor_clipped, rec_tensor_clipped, score_tensor_clipped = (
            clip_tensors_at_k(user_tensor, rec_tensor, score_tensor, k)
        )
        w_u_i_k = build_weight_tensor(
            user_tensor=user_tensor_clipped,
            item_tensor=rec_tensor_clipped,
            ratings_tensor=score_tensor_clipped,
            df=None,  # TODO: Gamb.
            weight_col=None,
            n_users=n_users,
            n_items=n_items,
        )
        sum_CEs_tensor += CE(
            weight_tensor=w_u_i_k, user_history_tensor=p_g_u, p_g_i=p_g_i
        )
    ACE = sum_CEs_tensor / N
    return ACE.mean().item()
