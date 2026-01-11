from calibrationUtils import clip_tensors_at_k
from constants import ITEM_COL, USER_COL
import numpy as np
import torch

from calibratedRecs.calibrationUtils import build_weight_tensor


def get_kl_divergence(
    dist_p: torch.Tensor, dist_q: torch.Tensor, epsilon: float = 1e-9
) -> float:
    """
    Calculates the KL divergence between two probability distributions represented as torch tensors.

    Parameters:
        dist_p (torch.Tensor): First probability distribution tensor.
        dist_q (torch.Tensor): Second probability distribution tensor.

    Returns:
        float: KL divergence value.
    """
    dist_p = dist_p + epsilon
    dist_q = dist_q + epsilon

    kl_div = torch.sum(dist_p * torch.log(dist_p / dist_q))
    return kl_div.item()


def CE(weight_tensor, user_history_tensor, p_g_i):
    q_g_u_k = weight_tensor @ p_g_i / weight_tensor.sum(dim=1, keepdim=True)
    q_g_u_k_filled = torch.nan_to_num(q_g_u_k, nan=0.0)
    return torch.abs(user_history_tensor - q_g_u_k_filled).mean(dim=1)


def mace(rec_df, p_g_u, p_g_i):

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rec_tensor = torch.tensor(rec_df[ITEM_COL].tolist(), device=dev).int()
    score_tensor = torch.tensor(
        rec_df["rating"].tolist(), dtype=torch.float32, device=dev
    )
    user_tensor = torch.tensor(rec_df[USER_COL].tolist(), device=dev).int()
    n_users = p_g_u.shape[0]
    n_items = p_g_i.shape[0]

    N = rec_tensor.shape[1]
    sum_CEs_tensor = torch.zeros(size=(n_users,), device=dev)
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

    not_nan_mask = ~torch.isnan(sum_CEs_tensor)
    filtered_tensor = sum_CEs_tensor[not_nan_mask]
    ACE = filtered_tensor / N
    #   TODO: um ponto em aberto para mim: Isso ta sendo calculado pra todo usuario de fato?
    # O .mean() Ta tirando uma media a nivel de genero ou a nivel de usuario?
    # Mas acho que ta tirando, sim. O proprio sum_CEs tensor tem tamanho n_users.
    return ACE.mean().item()
