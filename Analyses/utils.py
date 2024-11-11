import numpy as np
import pandas as pd

def date_to_t(date, start_date=pd.to_datetime('1970-01-01')):
    """
    Convert date to time index
    """
    date_time = pd.to_datetime(date)
    return (date_time - start_date).days

def t_to_date(t, start_date=pd.to_datetime('1970-01-01')):
    """
    Convert time index to date
    """
    return start_date + pd.DateOffset(days=t)

def real_to_p(number):
    """
    Map real number to (0,1) interval
    """
    return (np.tanh(number)+1)/2

def p_to_real(probability):
    """
    Map (0,1) interval to real number
    """
    return np.arctanh(2*probability-1)

def increment_to_vec(increment,length):
    """
    Convert increment to vector
    """
    return np.array([max(0,1-i*increment) for i in range(length)])

def to_increment(vec):
    """
    Convert vector to increment, or just return value
    """
    if isinstance(vec,np.ndarray):
        return 1-vec[1]
    else:
        return vec