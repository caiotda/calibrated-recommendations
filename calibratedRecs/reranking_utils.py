import torch
import pandas as pd
from calibratedRecs.constants import USER_COL, ITEM_COL, GENRE_COL

from calibration import Calibration

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rerank_by_calibration(
    recs, scores, ratings_df, n_users, n_items, calib_k, device=dev
):
    """
    Rerank recommendations by calibration.

    Args:
        recs (torch.Tensor): Recommendation indices of shape (n_users, k)
        scores (torch.Tensor): Recommendation scores of shape (n_users, k)
        ratings_df (pd.DataFrame): User history with columns [user, item, rating, timestamp, genre]
        n_users (int): Total number of users
        n_items (int): Total number of items
        calib_k (int): Number of top recommendations to consider for calibration
        device (torch.device): Device to place tensors on (default: cuda if available, else cpu)

    Returns:
        tuple: (reranked_recs, reranked_scores) both of shape (n_users, calib_k)
               where calib_k <= k represents the calibrated list length

    Notes:
        - Calibration adjusts recommendations to match genre distributions from user history
        - calib_k may be smaller than k for efficiency reasons, as calibration
          is often an expensive operation.
    """

    row_indices = list(range(recs.shape[0]))
    recs_ids_list = recs.cpu().tolist()
    recs_scores_list = scores.cpu().tolist()

    recs_df = pd.DataFrame(
        {
            "user": row_indices,
            "item": recs_ids_list,
            "rating": recs_scores_list,
        }
    )
    calibrator = Calibration(
        ratings_df=ratings_df,
        recommendation_df=recs_df,
        n_users=n_users,
        n_items=n_items,
    )

    calibrator.calibrate_for_users(k=calib_k)

    cali_df = calibrator.calibration_df.groupby(USER_COL).agg(
        {ITEM_COL: list, "rating": list}
    )
    reranked_recs = torch.tensor(cali_df[ITEM_COL].tolist(), device=device)
    reranked_recs_scores = torch.tensor(cali_df["rating"].tolist(), device=device)

    return reranked_recs, reranked_recs_scores
