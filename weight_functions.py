import pandas as pd
def get_linear_time_weight_rating(df):
    user_min_ts = df.groupby("user")["timestamp"].transform("min")
    user_max_ts = df.groupby("user")["timestamp"].transform("max")
    denom = user_max_ts - user_min_ts
    denom = denom.replace(0, 1)  
    df["w_twb"] = df["rating"] * (df["timestamp"] - user_min_ts) / denom

    return df

def get_rating_weight(df):
    df["w_rui"] = df["rating"]
    return df

def get_constant_weight(df):
    df["w_c"] = df["constant"]
    return df


#################### Recommmendation ones ###################################


def recommendation_twb_weighting(rec_df):
    return rec_df["top_k_rec_score"] * (1/rec_df["rank"])

def recommendation_score_weigthing(rec_df):
    return rec_df["top_k_rec_score"]
    
