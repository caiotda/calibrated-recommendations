from calibration import Calibration
import pickle


def instantiate_calibrator(rating_df, calibration_mode, distribution_mode, rec_df, READ_LOCALLY=False):
    calibrator_path = f"artifacts/calibrators/{distribution_mode}_distribution/{calibration_mode}_calibrator_bpr_mf_movielens_1m.pkl"

    #overwrite
    if (READ_LOCALLY):
            with open(calibrator_path, "rb") as f:
                calibrator = pickle.load(f)
            return calibrator
    else:
        calibrator = Calibration(rating_df, rec_df, calibration_mode, distribution_mode)
        calibrator.calibrate_for_users()
        with open(calibrator_path, "wb") as f:
            pickle.dump(calibrator, f)
    return calibrator