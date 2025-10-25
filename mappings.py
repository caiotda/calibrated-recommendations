
from calibratedRecs.weight_functions import (
    get_constant_weight,
    get_linear_time_weight_rating,
    get_rating_weight,
    recommendation_score_weigthing,
    recommendation_twb_weighting,
)


CALIBRATION_MODE_TO_DATA_PREPROCESS_FUNCTION = {
    "constant": get_constant_weight,
    "rating": get_rating_weight,
    "linear_time": get_linear_time_weight_rating
}



CALIBRATION_MODE_TO_RECOMMENDATION_PREPROCESS_FUNCTION = {
    "constant": recommendation_score_weigthing,
    "rating": recommendation_score_weigthing,
    "linear_time": recommendation_twb_weighting
}


CALIBRATION_MODE_TO_COL_NAME = {
    "constant": "w_c",
    "rating": "w_rui",
    "linear_time": "w_twb"
}