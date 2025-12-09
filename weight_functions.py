from constants import USER_COL

def get_linear_time_weight_rating(df):
    if "timestamp" not in df.columns:
        raise ValueError("Column timestamp not found in DataFrame.")
    user_min_ts = df.groupby(USER_COL)["timestamp"].transform("min")
    user_max_ts = df.groupby(USER_COL)["timestamp"].transform("max")
    denom = user_max_ts - user_min_ts
    denom = denom.replace(0, 1)  
    df["linear_time"] = df["rating"] * (df["timestamp"] - user_min_ts) / denom

    return df

#################### Recommmendation ones ###################################


def recommendation_twb_weighting(rec_df):
    return rec_df["top_k_rec_score"] * (1/rec_df["rank"])

def recommendation_score_weigthing(rec_df):
    return rec_df["top_k_rec_score"]
