import time
import numpy as np
from sklearn.metrics import mean_squared_log_error

################
#  DECORATORS  #
################

def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed


def RMSLE(y_true:np.ndarray, y_pred:np.ndarray) -> np.float64:
    """The Root Mean Squared Log Error (RMSLE) metric
    Notes
    -----

    Version
    -------
    specification : E.M. (v.1 07/04/2022)
    implementation : E.M. (v.1 07/04/2022)
    """
    return np.sqrt(mean_squared_log_error(y_true, y_pred))
