import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

# test_data = pd.read_sas('Data/Raw/KPSC/testing.sas7bdat')

# Function to categorize pathogens
def categorize_pathogens(pathogen):
    pathogen = str(pathogen).upper()
    
    if "PARAINFLUENZA" in pathogen:
        return "Parainfluenza"
    elif "INFLUENZA" in pathogen:
        return "Influenza"
    elif "RESPIRATORY SYNCYTIAL VIRUS" in pathogen:
        return "RSV"
    elif "METAPNEUMOVIRUS" in pathogen:
        return "HMPV"
    elif "ADENOVIRUS" in pathogen:
        return "Adenovirus"
    elif "SARS-COV-2" in pathogen:
        return "SARS-CoV-2"
    elif "ENTEROVIRUS" in pathogen:
        return "Enterovirus"
    else:
        return "Other"

def categorize_ndi(ndi):
    thresholds = np.array([-1,0,1])
    names = np.array(['Low', 'Medium-Low', 'Medium-High', 'High'])
    return names[np.digitize(ndi, thresholds)]

def categorize_age(age):
    thresholds = np.array([3, 12, 5*12, 18*12, 40*12, 65*12])
    names = np.array(['<3m', '3-11m', '1-4y', '5-17y', '18-39y', '40-64y', '>=65y'])
    return names[np.digitize(age, thresholds)]

def process_data(df):
    # convert YEAR, StudyID, lab_days, age, and age_in_mo to integers
    df['YEAR'] = df['YEAR'].astype(int)
    df['StudyID'] = df['StudyID'].astype(int)
    df['lab_days'] = df['lab_days'].astype(int)
    df['age'] = df['age'].astype(int)
    df['age_in_mo'] = df['age_in_mo'].astype(int)

    # Filter for PCR tests only
    df = df[df['lab_type'] == 'PCR'].copy()
    
    # Filter for valid results only
    df = df[df['result_val'] != 'Invalid'].copy()
    
    # Create binary result column (1 for positive, 0 for negative)
    df['is_positive'] = df['result_val'].apply(lambda x: 1 if x == 'Positive' else 0)

    # Pathogen, NDI, and age categorization
    df['pathogen_group'] = df['pathogen'].apply(categorize_pathogens)
    df['ndi_group'] = categorize_ndi(df['ndi'])
    df['age_group'] = categorize_age(df['age_in_mo'])

    # remove "Other" pathogens
    df = df[df['pathogen_group'] != "Other"].copy()

    # Time values
    # taking account of the fact that lab_days is the number of days since October 1st of that year, convert to days since start of year
    df['year_days'] = df['lab_days'] + 274
    # if lab_days is greater than 365, then it is in the next year
    df.loc[df['year_days'] > 365, 'YEAR'] = df.loc[df['year_days'] > 365,'YEAR'] + 1
    df.loc[df['year_days'] > 365, 'year_days'] = df.loc[df['year_days'] > 365,'year_days'] - 365
    # assign the date
    df['date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + (df['year_days']).astype(str), format='%Y-%j')
    # get the week of the year
    df['week'] = df['date'].dt.isocalendar().week
    # get the month of the year
    df['month'] = df['date'].dt.month

    # define year_week and year_month
    df['year_week'] = df['YEAR'].astype(str) + '-' + df['week'].astype(str)
    df['year_month'] = df['YEAR'].astype(str) + '-' + df['month'].astype(str)

    return df

# create a wide format dataset with one row per patient-test date for a given pathogen pair
def create_positivity_table(df,pathogen1,pathogen2,period,restrictive=False):
    # Filter for the two pathogens of interest
    # df = df[(df['pathogen_group'] == pathogen1) | (df['pathogen_group'] == pathogen2)].copy()
    if restrictive:
        index = ["StudyID", period, "date"]
        dfindex = ['StudyID', period, 'date', 'age_group', 'ndi_group']
    else:
        index = ["StudyID", period]
        dfindex = ['StudyID', period, 'age_group', 'ndi_group']
    # print(f"Creating positivity table for {pathogen1} and {pathogen2}...")
    # Create a new DataFrame with one row per patient-test period for a given pathogen pair
    # Use aggfunc='max' so that if any of the tests in the pathogen category in the period are positive, a positive is recorded
    df_wide = df.pivot_table(index=index, 
                             columns='pathogen_group', 
                             values='is_positive', 
                             aggfunc='max').reset_index()
    # remove rows with NAs
    # df_wide = df_wide.dropna()
    # add demographic information
    df_wide = df_wide.merge(df[dfindex].drop_duplicates(), on=index, how='left')
    
    return df_wide

# perform a Chochran-Mantel-Haenszel test, stratifying by age group and NDI group
def perform_cmh_test(positivity_table,pathogen1,pathogen2,period,threshold=0):
    # Create set of contingency tables for each age group and NDI group
    NAG = len(positivity_table['age_group'].unique())
    NNDIG = len(positivity_table['ndi_group'].unique())
    NP = len(positivity_table[period].unique())
    cmh_tables = np.zeros([NAG, NNDIG, NP, 2, 2])

    print(f"Creating {str(NAG*NNDIG*NP)} contingency tables...")
    stime = time.time()
    for i, age_group in enumerate(positivity_table['age_group'].unique()):
        for j, ndi_group in enumerate(positivity_table['ndi_group'].unique()):
            if i+j > 0:
                pdone = (i*NNDIG + j) / (NAG*NNDIG)
                time_remaining = (time.time() - stime) * (1 - pdone) / pdone
                print(f"{100*pdone:.0f}% complete, ETA {time_remaining:.0f}s      ", end='\r')
            for k, period_t in enumerate(positivity_table[period].unique()):
                # Create a contingency table for the current age group and NDI group
                results = positivity_table[(positivity_table['age_group'] == age_group) & 
                                                (positivity_table['ndi_group'] == ndi_group) &
                                                (positivity_table[period] == period_t)][[pathogen1, pathogen2]].values
                contingency_table = pd.crosstab(results[:, 0], results[:, 1]).values
                if contingency_table.shape == (2, 2) and np.sum(contingency_table) > threshold:
                    cmh_tables[i, j, k] = contingency_table
    
    # Collapse into age group and NDI group pairs
    cmh_tables = cmh_tables.reshape(-1, 2, 2)
    # Remove empty tables
    cmh_tables = cmh_tables[~np.all(cmh_tables == 0, axis=(1,2))]
    # # save to csv
    # cmh_tables_df = pd.DataFrame(cmh_tables.reshape(-1, 4), columns=['A', 'B', 'C', 'D'])
    # cmh_tables_df.to_csv('Data/Processed/cmh_tables.csv', index=False)
    
    # Calculate odds ratio
    D, C, B, A = cmh_tables[:, 0, 0], cmh_tables[:, 0, 1], cmh_tables[:, 1, 0], cmh_tables[:, 1, 1]
    T = np.sum(cmh_tables, axis=(1,2))
    R = np.sum(A*D/T)/np.sum(B*C/T)

    # calculate test statistic
    N1, N2, M1, M2 = A + B, C + D, A + C, B + D
    chi_CMH = np.square(np.sum(A - N1*M1/T)) / np.sum(N1*N2*M1*M2/(T**2 * (T - 1)))

    # find p-value corresponding to the test statistic
    p_value = stats.chi2.sf(chi_CMH, 1)

    return R, p_value



# test_data = pd.read_sas('Data/Raw/KPSC/testing.sas7bdat', encoding='utf-8')
# start_time = time.time()
# df = process_data(test_data)
# print(f"Data processed in {time.time() - start_time} seconds")
# # save the processed data
# df.to_csv('Data/Processed/testing.csv', index=False)
# print(df["pathogen_group"].value_counts())
# load the processed data
df = pd.read_csv('Data/Processed/testing.csv')

# # calculate overal positivity rate for respiratory pathogens
# pt = create_positivity_table(df,"A","B","year_month",restrictive=True)
# pt.to_csv("Data/Processed/positivity_table_all_pathogens_year_month_restrictive.csv", index=False)
# pt = pd.read_csv("Data/Processed/positivity_table_all_pathogens_year_month_restrictive.csv")
# test_results = pt[["RSV", "Influenza"]]
# # print(test_results)
# # print(test_results.max(axis=1))
# print(test_results.max(axis=1).mean())

pathogens_of_interest = ["RSV", "HMPV", "Adenovirus", "Influenza", "Parainfluenza", "SARS-CoV-2", "Enterovirus"]
results = []
for restrictive in [False, True]:
    periods = ['month','week','year_month','year_week','date'][2*(1-int(restrictive)):]
    for period in periods:
        for i, pathogen1 in enumerate(pathogens_of_interest):
            for pathogen2 in pathogens_of_interest[i+1:]:
                print(f"Testing {pathogen1} and {pathogen2}...")
                start_time = time.time()
                pt = create_positivity_table(df_random, pathogen1, pathogen2, period, restrictive=True)
                print(f"Positivity table created in {time.time() - start_time} seconds")
                # save the positivity table
                # pt.to_csv('Data/Processed/positivity_table_'+pathogen1+pathogen2+period+["","_restrictive"][restrictive]+'.csv', index=False)
                # # load the positivity table
                # pt = pd.read_csv('Data/Processed/positivity_table_'+pathogen1+pathogen2+period+'.csv')
                # perform the CMH test
                start_time = time.time()
                R, p_value = perform_cmh_test(pt, pathogen1, pathogen2, period,0)
                print(f"CMH test performed in {time.time() - start_time} seconds")
                print(f"Odds Ratio: {R}")
                print(f"P-value: {p_value}")
                results.append({
                    'match_by_date': restrictive,
                    'period': period,
                    'pathogen1': pathogen1,
                    'pathogen2': pathogen2,
                    'odds_ratio': R,
                    'p_value': p_value
                })
                # save the results
                results_df = pd.DataFrame(results)
                results_df.to_csv('Data/Processed/viral_interference_CMH_tests.csv', index=False)