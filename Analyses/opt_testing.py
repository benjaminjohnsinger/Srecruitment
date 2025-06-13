# params["BETA"] = 0.1
# def likelihood(pms):
#     pams = params.copy()
#     pams["BETA"] = pms
#     return -SIS_likelihood(noisy_incidence,pams,POINTS,STATE0,OBS_AGE,p_time_to_obs,age=True,incidence=True)
# time0 = time.time()
# opt = sp.optimize.minimize(likelihood,0.1,method='Nelder-Mead')
# time_1param = time.time()-time0
# print("BETA estimation time:",time_1param)
# print(opt)

# params["P_OBS"] = 0.02*jnp.ones(N_S)
# def likelihood(pms):
#     pams = params.copy()
#     pams["BETA"] = pms[0]
#     pams["P_OBS"] = pms[1]*jnp.ones(N_S)
#     return -SIS_likelihood(noisy_incidence,pams,POINTS,STATE0,OBS_AGE,p_time_to_obs,age=True,incidence=True)
# time0 = time.time()
# opt = sp.optimize.minimize(likelihood,[0.1,0.02],method='Nelder-Mead')
# time_2param = time.time()-time0
# print("BETA and P_OBS estimation time:",time_2param)
# print(opt)

# params["SEASONALITY"] = 0.05
# def likelihood(pms):
#     pams = params.copy()
#     pams["BETA"] = pms[0]
#     pams["P_OBS"] = pms[1]*jnp.ones(N_S)
#     pams["SEASONALITY"] = pms[2]
#     return -SIS_likelihood(noisy_incidence,pams,POINTS,STATE0,OBS_AGE,p_time_to_obs,age=True,incidence=True)
# time0 = time.time()
# opt = sp.optimize.minimize(likelihood,[0.1,0.02,0.05],method='Nelder-Mead')
# time_3param = time.time()-time0
# print("BETA, P_OBS and SEASONALITY estimation time:",time_3param)
# print(opt)

# params["OFFSET"] = 0.1
# def likelihood(pms):
#     pams = params.copy()
#     pams["BETA"] = pms[0]
#     pams["P_OBS"] = pms[1]*jnp.ones(N_S)
#     pams["SEASONALITY"] = pms[2]
#     pams["OFFSET"] = pms[3]
#     return -SIS_likelihood(noisy_incidence,pams,POINTS,STATE0,OBS_AGE,p_time_to_obs,age=True,incidence=True)
# time0 = time.time()
# opt = sp.optimize.minimize(likelihood,[0.1,0.02,0.05,0.1],method='Nelder-Mead')
# time_4param = time.time()-time0
# print("BETA, P_OBS, SEASONALITY and OFFSET estimation time:",time_4param)
# print(opt)

# params["WANE"] = 1/30*jnp.array([0.0,1.0,0.0])/365
# def likelihood(pms):
#     pams = params.copy()
#     pams["BETA"] = pms[0]
#     pams["P_OBS"] = pms[1]*jnp.ones(N_S)
#     pams["SEASONALITY"] = pms[2]
#     pams["OFFSET"] = pms[3]
#     pams["WANE"] = 1/30*jnp.array([0.0,pms[4],0.0])/365
#     return -SIS_likelihood(noisy_incidence,pams,POINTS,STATE0,OBS_AGE,p_time_to_obs,age=True,incidence=True)
# time0 = time.time()
# opt = sp.optimize.minimize(likelihood,[0.1,0.02,0.05,0.1,1.0],method='Nelder-Mead')
# time_5param = time.time()-time0
# print("BETA, P_OBS, SEASONALITY, OFFSET and WANE estimation time:",time_5param)
# print(opt)

# OBS_AGE = age_detection(NAG,0.4,0.85,0.1)
# def likelihood(pms):
#     pams = params.copy()
#     pams["BETA"] = pms[0]
#     pams["P_OBS"] = pms[1]*jnp.ones(N_S)
#     pams["SEASONALITY"] = pms[2]
#     pams["OFFSET"] = pms[3]
#     pams["WANE"] = 1/30*jnp.array([0.0,pms[4],0.0])/365
#     obs_age = age_detection(NAG,pms[5],0.85,0.1)
#     return -SIS_likelihood(noisy_incidence,pams,POINTS,STATE0,obs_age,p_time_to_obs,age=True,incidence=True)
# time0 = time.time()
# opt = sp.optimize.minimize(likelihood,[0.1,0.02,0.05,0.1,1.0,0.5],method='Nelder-Mead')
# time_6param = time.time()-time0
# print("BETA, P_OBS, SEASONALITY, OFFSET, WANE and OBS_AGE (young) estimation time:",time_6param)
# print(opt)

# OBS_AGE = age_detection(NAG,0.4,0.3,0.1)
# def likelihood(pms):
#     pams = params.copy()
#     pams["BETA"] = pms[0]
#     pams["P_OBS"] = pms[1]*jnp.ones(N_S)
#     pams["SEASONALITY"] = pms[2]
#     pams["OFFSET"] = pms[3]
#     pams["WANE"] = 1/30*jnp.array([0.0,pms[4],0.0])/365
#     obs_age = age_detection(NAG,pms[5],pms[6],0.1)
#     return -SIS_likelihood(noisy_incidence,pams,POINTS,STATE0,obs_age,p_time_to_obs,age=True,incidence=True)
# time0 = time.time()
# opt = sp.optimize.minimize(likelihood,[0.1,0.02,0.05,0.1,1.0,0.5,0.3],method='Nelder-Mead')
# time_7param = time.time()-time0
# print("BETA, P_OBS, SEASONALITY, OFFSET, WANE and OBS_AGE (young and old) estimation time:",time_7param)
# print(opt)

# OBS_AGE = age_detection(NAG,0.4,0.3,0.65)
# def likelihood(pms):
#     pams = params.copy()
#     pams["BETA"] = pms[0]
#     pams["P_OBS"] = pms[1]*jnp.ones(N_S)
#     pams["SEASONALITY"] = pms[2]
#     pams["OFFSET"] = pms[3]
#     pams["WANE"] = 1/30*jnp.array([0.0,pms[4],0.0])/365
#     obs_age = age_detection(NAG,pms[5],pms[6],pms[7])
#     return -SIS_likelihood(noisy_incidence,pams,POINTS,STATE0,obs_age,p_time_to_obs,age=True,incidence=True)
# time0 = time.time()
# opt = sp.optimize.minimize(likelihood,[0.1,0.02,0.05,0.1,1.0,0.5,0.3,0.65],method='Nelder-Mead')
# time_8param = time.time()-time0
# print("BETA, P_OBS, SEASONALITY, OFFSET, WANE and OBS_AGE (young and old and ratio) estimation time:",time_8param)
# print(opt)

# plt.plot([1,2,3,4,5,6,7,8],[time_1param,time_2param,time_3param,time_4param,time_5param,time_6param,time_7param,time_8param])
# plt.xlabel('Number of parameters')
# plt.ylabel('Time (s)')
# plt.title('Time taken to estimate parameters')
# plt.savefig('Figures/parameter_estimation_time.png',dpi=300)



# opt = sp.optimize.minimize(likelihood,[0.1,0.02],method='Nelder-Mead')
# print(opt)
# print(-likelihood(opt.x[0],opt.x[1]))

# fig, ax = plt.subplots(1,1,figsize=(6.5,6.5))
# im = ax.imshow([[-likelihood([beta,pobs]) for beta in jnp.linspace(0.75,0.85,10)] for pobs in jnp.linspace(0.009,0.011,10)],cmap='viridis')
# ax.set_xticks(jnp.linspace(0,9,5))
# ax.set_xticklabels(jnp.round(jnp.linspace(0.80,0.82,5),2))
# ax.set_yticks(jnp.linspace(0,9,5))
# ax.set_yticklabels(jnp.round(jnp.linspace(0.009,0.011,5),3))
# plt.colorbar(im)
# plt.xlabel('BETA')
# plt.ylabel('P_OBS')
# plt.title('Log likelihood of simulated noisy data for different values of BETA and P_OBS')
# plt.savefig('Figures/likelihood_BETA_P_OBS.png',dpi=300)

# plt.plot(jnp.linspace(0.01,0.1,50),[-likelihood(x) for x in jnp.linspace(0.01,0.1,50)])
# plt.scatter(opt.x,-likelihood(opt.x),color='red')
# plt.scatter(0.08,-likelihood(0.08),color='green')
# plt.xlabel('BETA')
# plt.ylabel('Log likelihood')
# plt.title('Log likelihood of simulated noisy data for different values of BETA')
# plt.savefig('Figures/likelihood_BETA.png',dpi=300)

# params["BETA"] = opt.x