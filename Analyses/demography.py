import pandas as pd
import matplotlib.pyplot as plt

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

# plot, with dates after 2022 in a different color, with legend and labels
fig, ax = plt.subplots(figsize=(6,6))
BIRTHS.plot(ax=ax,color="#648FFF")
BIRTHS.loc['2023':].plot(ax=ax,color="#DC267F")
plt.legend(['Resident births','Provisional data'])
ax.get_legend().get_lines()[0].set_color('#648FFF')
ax.get_legend().get_lines()[1].set_color('#DC267F')
# add detail panel to plot with data from 2016 onwards
axins = ax.inset_axes([0.42, 0.13, 0.42, 0.3])
BIRTHS.loc['2016':].plot(ax=axins,color="#648FFF")
BIRTHS.loc['2023':].plot(ax=axins,color="#DC267F")
axins.get_legend().remove()
axins.set_xticks(['2016-01-01','2020-01-01','2024-01-01'])
axins.set_yticks([])
ax.indicate_inset_zoom(axins)

ax.set_ylabel('Monthly births')
ax.set_xlabel('Date')
ax.set_title('Monthly births in California')
plt.tight_layout()
plt.savefig('Figures/births_CA.png',dpi=300)
