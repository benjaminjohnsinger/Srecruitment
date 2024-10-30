import numpy as np

def STATIC(t):
    return 1

def STEP(t,t_lockdown,duration,reduction):
    return 1-reduction if t < t_lockdown + duration and t > t_lockdown else 1

def RAMP(t,t_lockdown,duration,recovery_duration,reduction):
    return 1-reduction if t < t_lockdown + duration and t > t_lockdown else 1 + reduction*((t-t_lockdown-duration)/recovery_duration - 1) if t > t_lockdown and t < t_lockdown + duration + recovery_duration else 1

def piecewise(t,ts,fs):
    return fs[np.where(ts <= t)[0][-1]]