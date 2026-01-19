from distributions import  get_gleb_distribution

CALIBRATION_MODE_TO_RECOMMENDATION_COL = {
    "constant": "top_k_rec_id",
    "rating": "top_k_rec_id",
    "linear_time": "top_k_rec_id_cumulative_rank"
}

CALIBRATION_MODE_TO_COL_NAME = {
    "constant": "w_c",
    "rating": "w_rui",
    "linear_time": "w_twb"
}



DISTRIBUTION_MODE_TO_FUNCTION = {
    'steck': None,
    'gleb': get_gleb_distribution
}



def validate_modes(weight, distribution_mode):
    """
    Validates if the weight strategy and genre distribution mode
    have been implemented
    """
    if weight not in CALIBRATION_MODE_TO_RECOMMENDATION_COL:
        raise ValueError(
            f"Invalid weight mode: {weight}. Must be one of {list(CALIBRATION_MODE_TO_RECOMMENDATION_COL.keys())}"
        )
    if distribution_mode not in DISTRIBUTION_MODE_TO_FUNCTION:
        raise ValueError(
            f"Invalid distribution mode: {distribution_mode}. Must be one of {list(DISTRIBUTION_MODE_TO_FUNCTION.keys())}"
        )