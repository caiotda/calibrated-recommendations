
def get_linear_time_weight_rating(df):
    user_min_ts = df.groupby("user")["timestamp"].transform("min")
    user_max_ts = df.groupby("user")["timestamp"].transform("max")
    denom = user_max_ts - user_min_ts
    denom = denom.replace(0, 1)  
    df["w_twb"] = df["rating"] * (df["timestamp"] - user_min_ts) / denom

    return df

def get_rating_weight(df):
    df["w_r_u_i"] = df["rating"]
    return df

def get_constant_weight(df):
    df["w_c"] = 1
    return df

