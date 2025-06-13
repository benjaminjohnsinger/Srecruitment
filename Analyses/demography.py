import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
from numba import jit
from utils import date_to_t, t_to_date

## BIRTHS
# load births data
BIRTHS = pd.read_csv("Data/Raw/California_births_1960_2022.csv", delimiter=",")
# use Year column and Month column to create a datetime index
BIRTHS.index = pd.to_datetime(BIRTHS["Year"].astype(str) + "-" + BIRTHS["Month"].astype(str), format="%Y-%m")
# filter Geography_Type="Residence" and Strata_Name="Total Population"
BIRTHS = BIRTHS.loc[(BIRTHS["Geography_Type"] == "Residence") & (BIRTHS["Strata_Name"] == "Total Population")]
# drop all columns other than Count
BIRTHS = BIRTHS.drop(columns=["Geography_Type","Strata","Strata_Name","Year","Month","Annotation_Code","Annotation_Desc","Data_Revision_Date"])
# rename Count column to Births
BIRTHS = BIRTHS.rename(columns={"Count":"Births"})

# add provisional births from 2023 to 2024 - "Occurence" data has correlation coefficient 0.9999904980205537 with "Residence" data
BIRTHS_2023_2024 = pd.read_csv("Data/Raw/California_births_2023_2024_provisional.csv", delimiter=",")
BIRTHS_2023_2024.index = pd.to_datetime(BIRTHS_2023_2024["Year"].astype(str) + "-" + BIRTHS_2023_2024["Month"].astype(str), format="%Y-%m")
BIRTHS_2023_2024 = BIRTHS_2023_2024.loc[BIRTHS_2023_2024["Strata_Name"] == "Total Population"]
BIRTHS_2023_2024 = BIRTHS_2023_2024.drop(columns=["Geography_Type","Strata","Strata_Name","Year","Month","Annotation_Code","Annotation_Desc","Data_Revision_Date"])
BIRTHS_2023_2024 = BIRTHS_2023_2024.rename(columns={"Count":"Births"})
BIRTHS = pd.concat([BIRTHS, BIRTHS_2023_2024])
# reformat births index to be number of days since '1970-01-01'
BIRTHS.index = (BIRTHS.index - pd.to_datetime("1970-01-01")).days

## POPULATION
# load population data
POPULATION = pd.read_csv("Data/Raw/California_population_1900_2023.csv", delimiter=",")
# use Date column to create a datetime index based on YYYY-MM-DD format
POPULATION.index = pd.to_datetime(POPULATION["Date"], format="%Y-%m-%d")
# drop Date column
POPULATION = POPULATION.drop(columns=["Date","Annual % Change"])
# reformat population index to be number of days since 1970-01-01
POPULATION.index = (POPULATION.index - pd.to_datetime("1970-01-01")).days


# BIRTHS and POPULATION as numpy vectors
BIRTHS_IDX =jnp.array(BIRTHS.index)
POPULATION_IDX = jnp.array(POPULATION.index)
BIRTHS_NP = jnp.array(BIRTHS["Births"])
POPULATION_NP = jnp.array(POPULATION["Population"])
# print(POPULATION_NP)

def births(t):
    """
    Return number of births per day at time t, with t the number of days since 1970-01-01
    """
    return BIRTHS.iloc[(BIRTHS.index<=t).argmin()]["Births"]/30.44

# @jit
def birth_rate(t, BIRTHS_NP=BIRTHS_NP, BIRTHS_IDX=BIRTHS_IDX, POPULATION_NP=POPULATION_NP, POPULATION_IDX=POPULATION_IDX):
    """
    Return number of births per person per day at time t, with t the number of days since 1970-01-01
    """
    bths = BIRTHS_NP[jnp.argmax(BIRTHS_IDX>=t)]/30.44
    pop = POPULATION_NP[jnp.argmax(POPULATION_IDX>=t)]
    # if t>BIRTHS_IDX[-1] or t>POPULATION_IDX[-1]:
    #     bths = BIRTHS_NP[-1]/30.44
    #     pop = POPULATION_NP[-1]
    out_of_bounds = (t > BIRTHS_IDX[-1]) | (t > POPULATION_IDX[-1])
    bths = jnp.where(out_of_bounds, BIRTHS_NP[-1]/30.44, bths)
    pop = jnp.where(out_of_bounds, POPULATION_NP[-1], pop)
    return bths/pop



# PERIOD = pd.date_range(start=pd.to_datetime('2000-10-01'), end=pd.to_datetime('2024-10-01'), freq='D')

# POINTS = jnp.array(date_to_t(PERIOD))

# plt.plot([t_to_date(pt) for pt in POINTS],[birth_rate(pt) for pt in POINTS])
# plt.show()

# # plot, with dates after 2022 in a different color, with legend and labels
# fig, ax = plt.subplots(figsize=(6,6))
# BIRTHS.plot(ax=ax,color="#648FFF")
# BIRTHS.loc[(pd.to_datetime('2023-01-01')-pd.to_datetime('1970-01-01')).days:].plot(ax=ax,color="#DC267F")
# plt.legend(['Resident births','Provisional data'])
# ax.get_legend().get_lines()[0].set_color('#648FFF')
# ax.get_legend().get_lines()[1].set_color('#DC267F')
# # add detail panel to plot with data from 2016 onwards
# axins = ax.inset_axes([0.42, 0.13, 0.42, 0.3])
# BIRTHS.loc[(pd.to_datetime('2016-01-01')-pd.to_datetime('1970-01-01')).days:].plot(ax=axins,color="#648FFF")
# BIRTHS.loc[(pd.to_datetime('2023-01-01')-pd.to_datetime('1970-01-01')).days:].plot(ax=axins,color="#DC267F")
# axins.get_legend().remove()
# axins.set_xticks([(pd.to_datetime('2016-01-01')-pd.to_datetime('1970-01-01')).days,(pd.to_datetime('2020-01-01')-pd.to_datetime('1970-01-01')).days,(pd.to_datetime('2024-01-01')-pd.to_datetime('1970-01-01')).days])
# axins.set_yticks([])
# # axins.set_ylim(0,40000)
# ax.indicate_inset_zoom(axins)

# # years as x ticks - every 10 years
# years = pd.date_range(start='1960-01-01', end='2025-01-01', freq='10YE')
# # years = years[years.year!=2020]
# ax.set_xticks((years-pd.to_datetime('1970-01-01')).days)
# ax.set_xticklabels(years.year)
# # and for axins, restricting to range from 2016 to 2024
# years = pd.date_range(start='2016-01-01', end='2025-01-01', freq='2YE')
# axins.set_xticks((years-pd.to_datetime('1970-01-01')).days)
# axins.set_xticklabels(years.year)

# ax.set_ylabel('Monthly births')
# ax.set_xlabel('Date')
# ax.set_title('Monthly births in California')
# ax.set_ylim(0,55250)
# plt.tight_layout()
# plt.savefig('Figures/births_CA.png',dpi=300)
