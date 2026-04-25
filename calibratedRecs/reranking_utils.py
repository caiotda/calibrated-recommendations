import torch
import pandas as pd
from calibratedRecs.constants import USER_COL, ITEM_COL, GENRE_COL

from calibratedRecs.calibration import Calibration
from calibratedRecs.constants import UNKNOWN_GENRE

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rerank_by_calibration(
    recs,
    scores,
    ratings_df,
    n_users,
    n_items,
    calib_k,
    item2genreMap,
    calibration_type,
    device=dev,
    dist_function="kl",
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
        item2genremap (dict): Mapping from item IDs to their genres
        device (torch.device): Device to place tensors on (default: cuda if available, else cpu)
        calibration_type (str): Type of calibration to perform (e.g, linear based, constant)
        dist_function (Str: either kl or hellinger): Type of distance function between probabilities to use
            (defaults to kl divergence)

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

    recs_df = recs_df.explode(["item", "rating"])
    recs_df["genres"] = (
        recs_df["item"]
        .map(item2genreMap)
        .apply(lambda x: x if isinstance(x, list) else [UNKNOWN_GENRE])
    )

    calibrator = Calibration(
        ratings_df=ratings_df,
        recommendation_df=recs_df,
        n_users=n_users,
        n_items=n_items,
        weight=calibration_type,
        prob_dist_function=dist_function,
    )

    calibrator.calibrate_for_users(k=calib_k, verbose=False)

    cali_df = calibrator.calibration_df.groupby(USER_COL).agg(
        {ITEM_COL: list, "rating": list}
    )
    reranked_recs = torch.tensor(cali_df[ITEM_COL].tolist(), device=device)
    reranked_recs_scores = torch.tensor(cali_df["rating"].tolist(), device=device)

    return reranked_recs, reranked_recs_scores
