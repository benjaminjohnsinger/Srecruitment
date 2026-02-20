import pandas as pd
import numpy as np

from utils import date_to_t

from Parameters.times_and_contacts import EPOCH, END
FULL_POINTS = np.array(date_to_t(pd.date_range(start=EPOCH, end=END, freq='D')), dtype=np.float32)

## BIRTH RATE
# load births data
BIRTHS = pd.read_csv("Data/Raw/California_births_1960_2023.csv", delimiter=",")
# use Year column and Month column to create a datetime index
BIRTHS.index = pd.to_datetime(BIRTHS["Year"].astype(str) + "-" + BIRTHS["Month"].astype(str), format="%Y-%m")
# filter Geography_Type="Residence" and Strata_Name="Total Population"
BIRTHS = BIRTHS.loc[(BIRTHS["Geography_Type"] == "Residence") & (BIRTHS["Strata_Name"] == "Total Population")]
# drop all columns other than Count
BIRTHS = BIRTHS.drop(columns=["Geography_Type","Strata","Strata_Name","Year","Month","Annotation_Code","Annotation_Desc","Data_Revision_Date"])
# rename Count column to Births
BIRTHS = BIRTHS.rename(columns={"Count":"Births"})

# add provisional births from 2024 to 2025 - "Occurence" data has correlation coefficient 0.9999904980205537 with "Residence" data
BIRTHS_2024_2025 = pd.read_csv("Data/Raw/California_births_2024_2025_provisional.csv", delimiter=",")
BIRTHS_2024_2025.index = pd.to_datetime(BIRTHS_2024_2025["Year"].astype(str) + "-" + BIRTHS_2024_2025["Month"].astype(str), format="%Y-%m")
BIRTHS_2024_2025 = BIRTHS_2024_2025.loc[BIRTHS_2024_2025["Strata_Name"] == "Total Population"]
BIRTHS_2024_2025 = BIRTHS_2024_2025.drop(columns=["Geography_Type","Strata","Strata_Name","Year","Month","Annotation_Code","Annotation_Desc","Data_Revision_Date"])
BIRTHS_2024_2025 = BIRTHS_2024_2025.rename(columns={"Count":"Births"})
BIRTHS = pd.concat([BIRTHS, BIRTHS_2024_2025])
# reformat births index to be number of days since '1970-01-01'
BIRTHS.index = (BIRTHS.index - pd.to_datetime("1970-01-01")).days

# load population data
POPULATION = pd.read_csv("Data/Raw/California_population_1900_2024.csv", delimiter=",")
# use Date column to create a datetime index based on YYYY-MM-DD format
POPULATION.index = pd.to_datetime(POPULATION["Date"], format="%Y-%m-%d")
# drop Date column
POPULATION = POPULATION.drop(columns=["Date","Annual % Change"])
# reformat population index to be number of days since 1970-01-01
POPULATION.index = (POPULATION.index - pd.to_datetime("1970-01-01")).days

# BIRTHS and POPULATION as numpy vectors
BIRTHS_IDX =np.array(BIRTHS.index)
POPULATION_IDX = np.array(POPULATION.index)
BIRTHS_NP = np.array(BIRTHS["Births"])
POPULATION_NP = np.array(POPULATION["Population"])

# interplolation
BIRTHS_interp = np.interp(FULL_POINTS, BIRTHS_IDX, BIRTHS_NP)/30.44
POPULATION_interp = np.interp(FULL_POINTS, POPULATION_IDX, POPULATION_NP)
BIRTH_RATE = BIRTHS_interp/POPULATION_interp

np.savetxt("Data/Processed/birth_rate_daily.csv", BIRTH_RATE, delimiter=",")

## IMPORT
FLIGHTS = pd.read_csv('Data/Raw/Los_Angeles_International_Airport_-_Passenger_Traffic_By_Terminal.csv', delimiter=',')

MONTHLY_ARRIVALS = FLIGHTS.loc[FLIGHTS['Arrival_Departure'] == 'Arrival'].groupby('ReportPeriod').sum('Passenger_Count')
MONTHLY_ARRIVALS.index = pd.to_datetime(MONTHLY_ARRIVALS.index, format='%m/%d/%Y %H:%M:%S %p')
MONTHLY_ARRIVALS = MONTHLY_ARRIVALS.sort_index()
MONTHLY_ARRIVALS.index = (MONTHLY_ARRIVALS.index - pd.to_datetime('1970-01-01')).days
PASSENGER_COUNT = np.asarray(MONTHLY_ARRIVALS['Passenger_Count'])

ARRIVALS_IDX = np.array(MONTHLY_ARRIVALS.index)
MONTHLY_ARRIVALS_NP = np.array(PASSENGER_COUNT)

# interpolation
ARRIVALS = np.interp(FULL_POINTS, ARRIVALS_IDX, MONTHLY_ARRIVALS_NP)/30.44

# for days before first ARRIVALS_IDX, repeat first year of data, and for days after last ARRIVALS_IDX, repeat last year of data
for i in range(len(FULL_POINTS)):
    if FULL_POINTS[i] < ARRIVALS_IDX[0]:
        ARRIVALS[i] = ARRIVALS[ARRIVALS_IDX[0] + i%365]
    elif FULL_POINTS[i] > ARRIVALS_IDX[-1]:
        ARRIVALS[i] = ARRIVALS[ARRIVALS_IDX[-1] - 365 + i%365]

np.savetxt("Data/Processed/arrivals_daily.csv", ARRIVALS, delimiter=",")

## REGIONAL POSITIVITY
for first_idx, pathogen, filename in zip([14883,10500,10500,14883,14883,14883],["RSV","InfluenzaA","InfluenzaB","Metapneumovirus","Parainfluenza3","Adenovirus"],["RSV_PercentPositive_Regions","FluView_PercentPositive_Regions_A","FluView_PercentPositive_Regions_B","NREVSS_PCR_PercentPositive_MPV","NREVSS_PCR_PercentPositive_PIV3","NREVSS_PCR_PercentPositive_AdV"]):
    PP = pd.read_csv("Data/Processed/"+filename+".csv")
    PP.index = pd.to_datetime(PP["Date"], format="%Y-%m-%d")
    PP.index = (PP.index - pd.to_datetime("1970-01-01")).days
    PP_NP = np.array(PP[['Region '+str(i) for i in range(1,9)] + ['Region 10']].mean(axis=1))/100
    PP_IDX = np.array(PP.index)

    # interpolation
    POSITIVITY = np.interp(FULL_POINTS, PP_IDX, PP_NP)

    # for days before first IDX, repeat first year of data
    for i in range(len(FULL_POINTS)):
        if FULL_POINTS[i] < first_idx:
            POSITIVITY[i] = POSITIVITY[first_idx + i%365]
        else:
            break

    np.savetxt("Data/Processed/"+pathogen+"_positivity_daily.csv", POSITIVITY, delimiter=",")

## POPULATION
from Parameters.census_population import AGING_RATE
from Parameters.census_population import CENSUS_AGE_POP
from SISn_ODEs import single_pathogen_deltas as sis_deltas
import scipy as sp
from fit_MCMC import run_simulation
from utils import x_to_params
import jax.numpy as jnp
CONTACT_MATRIX = jnp.asarray(pd.read_csv('Data/Processed/contact_matrices/KP_contact_all_US_Census.csv', delimiter=',', header=None).values)
BETA = 0
WANE = np.zeros(3)
S_REL = np.ones(3)
P_OBS = np.ones(3)
OBS_AGE = np.ones(7)
RELATIVE_CONTACT = np.ones(FULL_POINTS.shape[0])
VAX_RATE = np.zeros((FULL_POINTS.shape[0],7))
MATERNAL_IMMUNITY = 0
REC_UP = np.zeros(3)
REC_SAME = np.zeros(3)
IMPORT_STRENGTH = np.zeros(FULL_POINTS.shape[0])
params = (FULL_POINTS, AGING_RATE, BIRTH_RATE, CONTACT_MATRIX,
            BETA, WANE, S_REL, P_OBS, OBS_AGE, RELATIVE_CONTACT, VAX_RATE, MATERNAL_IMMUNITY,
            REC_UP, REC_SAME, IMPORT_STRENGTH)

STATE0 = jnp.zeros((2*3+1,7))
STATE0 = STATE0.at[0,:].set(CENSUS_AGE_POP)
STATE0 = STATE0.flatten()
STATE0 = jnp.concatenate((jnp.array([0]), STATE0))

joints = jnp.array(FULL_POINTS)
values = run_simulation(params, STATE0, joints[-1], joints)
age_pops = np.array([jnp.sum(values[range(1+i,3*2*7+1,7),:],axis=0) for i in range(7)]).T

np.savetxt("Data/Processed/age_pops_daily.csv", age_pops, delimiter=",")


import matplotlib.pyplot as plt
from plotting import hsv_colors
for i in range(7):
    plt.plot(FULL_POINTS[16500:], age_pops[16500:,i], color=hsv_colors[i], label=f'Age group {i}')
plt.savefig("Figures/age_pops_daily.pdf")
