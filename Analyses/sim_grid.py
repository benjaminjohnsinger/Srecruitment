import jax.numpy as np
import scipy as sp
import itertools as it
from SISn_ODEs import single_pathogen_deltas as deltas_SIS

def sim_grid(state0,params,points,T_LOCKDOWN,LOCKDOWN_DURATION,
            grid_params=(("BETA","REC"),("WANE_UP","WANE_SAME")),grid_mode=("scale","scale"),N=25,factors=(1,1),deltas=deltas_SIS):
    N_params = len(grid_params)
    results = {}
    params_dict = {}
    for p_n in it.product(range(N),repeat=N_params):
        if all([p==0 for p in p_n[1:]]):
            print(p_n)
        # Scale parameters for exploration
        params_n = params.copy()
        for i,p in enumerate(p_n):
            for pname in grid_params[i]:
                if grid_mode[i] == "scale":
                    params_n[pname] = params[pname]*(1+(p/N-1/2))**factors[i]
                elif grid_mode[i] == "fade_vec":
                    vec_len = len(params[pname])
                    vec = np.array([(1-j*(p/(N*(vec_len-1))))**factors[i] for j in range(vec_len)])
                    vec = vec.reshape(params[pname].shape)
                    params_n[pname] = vec
                elif grid_mode[i] == "power_vec":
                    vec_len = len(params[pname])
                    vec = np.array([(1-(p/N))**(j*factors[i]) for j in range(vec_len)])
                    vec = vec.reshape(params[pname].shape)
                    params_n[pname] = vec
                elif grid_mode[i] == "based_vec":
                    vec_len = len(params[pname])
                    base_value = params[pname][0]-params[pname][1]
                    vec = np.array([(1-j*(base_value+(1/(vec_len-1)-base_value)*p/N))**factors[i] for j in range(vec_len)])
                    vec = vec.reshape(params[pname].shape)
                    params_n[pname] = vec
        # Run simulation
        result = sp.integrate.solve_ivp(deltas, (points[0],points[-1]), state0, method='RK45', t_eval=points, args=params_n.values())
        results[p_n] = result
        params_dict[p_n] = params_n
    return(params_dict,results)