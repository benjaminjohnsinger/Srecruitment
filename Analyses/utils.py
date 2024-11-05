import numpy as np
import pandas as pd

def date_to_t(date, start_date=pd.to_datetime('1970-01-01')):
    """
    Convert date to time index
    """
    date_time = pd.to_datetime(date)
    return (date_time - start_date).days