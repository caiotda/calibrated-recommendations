from calibratedRecs.constants import USER_COL, TIME_COL


def get_linear_time_weight_rating(df):
    if TIME_COL not in df.columns:
        raise ValueError(f"Column {TIME_COL} not found in DataFrame.")
    user_min_ts = df.groupby(USER_COL)[TIME_COL].transform("min")
    user_max_ts = df.groupby(USER_COL)[TIME_COL].transform("max")
    denom = user_max_ts - user_min_ts
    denom = denom.replace(0, 1)
    df["linear_time"] = (df[TIME_COL] - user_min_ts) / denom
    df["linear_time"] = df["linear_time"].fillna(0)
    return df


#################### Recommmendation ones ###################################


def recommendation_twb_weighting(rec_df):
    return rec_df["top_k_rec_score"] * (1 / rec_df["rank"])


def recommendation_score_weigthing(rec_df):
    return rec_df["top_k_rec_score"]
