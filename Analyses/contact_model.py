import numpy as np

def STATIC(t):
    return 1

def STEP(t,t_lockdown,duration,reduction):
    return 1-reduction if t < t_lockdown + duration and t > t_lockdown else 1