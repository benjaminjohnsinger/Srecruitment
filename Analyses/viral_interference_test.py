import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

# adjust plot text size
plt.rcParams.update({'font.size':20})
# text type is palatino
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Palatino']

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
    if pathogen1 != None and pathogen2 != None:
        df = df[(df['pathogen_group'] == pathogen1) | (df['pathogen_group'] == pathogen2)].copy()
    if restrictive:
        if period == 'date':
            index = ["StudyID", "date"]
            dfindex = ['StudyID', "date", 'age_group', 'ndi_group']
        else:
            index = ["StudyID", period, "date"]
            dfindex = ['StudyID', period, 'date', 'age_group', 'ndi_group']
    else:
        index = ["StudyID", period]
        dfindex = ['StudyID', period, 'age_group', 'ndi_group']
    print(f"Creating positivity table for {pathogen1} and {pathogen2}...")
    # Create a new DataFrame with one row per patient-test period for a given pathogen pair
    # Use aggfunc='max' so that if any of the tests in the pathogen category in the period are positive, a positive is recorded
    df_wide = df.pivot_table(index=index, 
                             columns='pathogen_group', 
                             values='is_positive', 
                             aggfunc='max').reset_index()
    # remove rows with NAs
    if pathogen1 != None and pathogen2 != None:
        df_wide = df_wide.dropna()
    # add demographic information
    df_wide = df_wide.merge(df[dfindex].drop_duplicates(), on=index, how='left')
    
    return df_wide

# perform a Chochran-Mantel-Haenszel test, stratifying by age group and NDI group
def perform_cmh_test(positivity_table,pathogen1,pathogen2,period,prevalence_correction1=1,prevalence_correction2=1,threshold=0):
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
    print("Overall counts:",np.sum(A), np.sum(B), np.sum(C), np.sum(D))
    print("Prevalence:", np.sum(A + B) / np.sum(A + B + C + D), np.sum(A + C) / np.sum(A + B + C + D))
    if prevalence_correction1 != 1 or prevalence_correction2 != 1:
        ab = prevalence_correction1 / prevalence_correction2
        C = ab*C + (ab - 1)*A
        D = D + (1 - ab)*(A + C) + (prevalence_correction1 - 1)*(A + B + C + D)
    print("Corrected prevalence:", np.sum(A + B) / np.sum(A + B + C + D), np.sum(A + C) / np.sum(A + B + C + D))
    T = A + B + C + D
    R = np.sum(A*D/T)/np.sum(B*C/T)

    # calculate test statistic
    N1, N2, M1, M2 = A + B, C + D, A + C, B + D
    chi_CMH = np.square(np.sum(A - N1*M1/T)) / np.sum(N1*N2*M1*M2/(T**2 * (T - 1)))

    # find p-value corresponding to the test statistic
    p_value = stats.chi2.sf(chi_CMH, 1)

    return R, p_value

def generate_tables(pathogens_of_interest,proportion_symptomatic=[],restrictive=False,period='year_month',load=False,threshold=0):
    results = []
    for i, pathogen1 in enumerate(pathogens_of_interest):
        for pathogen2 in pathogens_of_interest[i+1:]:
            print(f"Testing {pathogen1} and {pathogen2}...")
            if load:
                # load the positivity table
                pt = pd.read_csv('Data/Processed/positivity_table_'+pathogen1+pathogen2+period+["","_restrictive"][restrictive]+'.csv')
            else:
                start_time = time.time()
                pt = create_positivity_table(df, pathogen1, pathogen2, period, restrictive=restrictive)
                print(f"Positivity table created in {time.time() - start_time} seconds")
                # save the positivity table
                pt.to_csv('Data/Processed/positivity_table_'+pathogen1+pathogen2+period+["","_restrictive"][restrictive]+'.csv', index=False)
            # prevalence adjustments
            pc1 = 1/proportion_symptomatic[pathogen1] if pathogen1 in proportion_symptomatic else 1
            pc2 = 1/proportion_symptomatic[pathogen2] if pathogen2 in proportion_symptomatic else 1
            # perform the CMH test
            start_time = time.time()
            R, p_value = perform_cmh_test(pt, pathogen1, pathogen2, period,
                prevalence_correction1=pc1, prevalence_correction2=pc2, threshold=threshold)
            print(f"CMH test performed in {time.time() - start_time} seconds")
            print(f"Odds Ratio: {R}")
            print(f"P-value: {p_value}")
            # save
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
            results_df.to_csv('Data/Processed/viral_interference_CMH_tests' + ["", "_prevalence_correction"][proportion_symptomatic != []] + ["", "_restrictive"][restrictive] + "_threshold" + str(threshold) + '.csv', index=False)

# plot heatmap of odds ratios
def plot_heatmap(ax, results, pathogens_of_interest, title='Viral Interference Heatmap', significance=None):
    # Create a pivot table for the heatmap
    pivot_table = results.pivot(index='pathogen1', columns='pathogen2', values='odds_ratio')
    p_values = results.pivot(index='pathogen1', columns='pathogen2', values='p_value')

    pivot_table = pivot_table.reindex(pathogens_of_interest, axis=0).reindex(pathogens_of_interest, axis=1)
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='viridis_r', ax=ax, cbar_kws={'label': 'Odds Ratio'}, vmin=0.1, vmax=1)
    cells = ax.get_children()
    if significance != None:
        for i,p1 in enumerate(pathogens_of_interest):
            for j,p2 in enumerate([pathogen for pathogen in pathogens_of_interest if pathogen != p1]):
                if pivot_table.loc[p1,p2] < 1 and p_values.loc[p1,p2] < significance:
                    cell_color = cells[i * (len(pathogens_of_interest)-1) + j + 1].get_color()
                    ax.text(j + int(j>=i) + 0.8, i + 0.3, '*', color=cell_color, fontsize=30, ha='center', va='center')
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')
    # in tick labels, use abbreviations for pathogens - Adenovirus = AdV, Influenza = Flu, Parainfluenza = PIV, RSV = RSV, HMPV = HMPV
    pathogen_abbreviations = {
        "Adenovirus": "AdV",
        "Influenza": "Flu",
        "Parainfluenza": "PIV",
        "RSV": "RSV",
        "HMPV": "hMPV"}
    ax.set_xticklabels([pathogen_abbreviations[p.get_text()] for p in ax.get_xticklabels()], rotation=45, ha='right')
    ax.set_yticklabels([pathogen_abbreviations[p.get_text()] for p in ax.get_yticklabels()])


pathogens_of_interest = ["RSV", "HMPV", "Adenovirus", "Influenza", "Parainfluenza"]
proportion_symptomatic = {"All": 0.2894586894586895,
"Influenza": 0.5356125356125356,
"RSV": 0.341880341880342,
"Parainfluenza": 0.2279202279202281,
"HMPV": 0.6792022792022792,
"Rhinovirus": 0.2529914529914531,
"Adenovirus": 0.19601139601139594,
"Coronavirus": 0.2780626780626781} # Galanti et al. 2019 Epidemiology & Infection

if __name__ == "__main__":
    # test_data = pd.read_sas('Data/Raw/KPSC/testing.sas7bdat', encoding='utf-8')
    # test_data = test_data.loc[(test_data["lab_type"]=="PCR") & (test_data["pathogen"] != "SARS-COV-2 (COVID-19)")].copy()
    # # ((test_data["pathogen"] == "RESPIRATORY SYNCYTIAL VIRUS") | (test_data["pathogen"] == "RESPIRATORY SYNCYTIAL VIRUS SUBTYPE A") | (test_data["pathogen"] == "RESPIRATORY SYNCYTIAL VIRUS SUBTYPE B"))].copy()
    #     # ((test_data["pathogen"] == "INFLUENZA VIRUS A") | (test_data["pathogen"] == "INFLUENZA VIRUS B") | (test_data["pathogen"] == "INFLUENZA VIRUS A H1N1 2009") | (test_data["pathogen"] == "INFLUENZA VIRUS A SUBTYPE H1") | (test_data["pathogen"] == "INFLUENZA VIRUS A SUBTYPE/HEMAGGLUTININ H3") | (test_data["pathogen"] == "INFLUENZA A VIRUS") | (test_data["pathogen"] == "INFLUENZA VIRUS A+B"))]
    # N_T = len(test_data["StudyID"].unique())
    # print(N_T, "patients with non-covid PCR tests")
    # clinical_data = pd.read_sas('Data/Raw/KPSC/clinical_20241202.sas7bdat', encoding='utf-8')
    # print(len(clinical_data["StudyID"].unique()), "patients in the clinical dataset")
    # clinical_data["Date"] = pd.to_datetime(clinical_data["YEAR"].astype(int).astype(str) + '-10-01') + pd.to_timedelta(clinical_data["dx_days"],unit='D')

    # clinical_tests = test_data[test_data["StudyID"].isin(clinical_data["StudyID"])]
    # clinical_tests = clinical_tests[clinical_tests["lab_type"] == "PCR"].copy()
    # # clinical_data = clinical_data[clinical_data["setting"] == "Hospital admission"].copy()
    # clinical_data = clinical_data[clinical_data["StudyID"].isin(clinical_tests["StudyID"])]
    # print(clinical_data["setting"].value_counts())
    # print(len(clinical_data["StudyID"].unique()), "hospitalized patients in the clinical dataset with non-covid PCR tests")
    # clinical_tests["Date"] = pd.to_datetime(clinical_tests["YEAR"].astype(int).astype(str) + '-10-01') + pd.to_timedelta(clinical_tests["lab_days"],unit='D')
    # print('Y')
    # clinical_tests = clinical_data.merge(clinical_tests[["StudyID","Date"]],on="StudyID",how="left")
    # print('A')
    # clinical_tests = clinical_tests.rename(columns={"Date_x":"Clinical date","Date_y":"Test date"})
    # print('B')
    # matched_tests = clinical_tests[np.abs((pd.to_datetime(clinical_tests["Clinical date"]) - pd.to_datetime(clinical_tests["Test date"])).dt.days) <= 14]
    # print(matched_tests.head())
    # print(len(matched_tests), "non-covid PCR tests correspond to a hospital admission within 14 days")
    # # hospital_tests = matched_tests[matched_tests["setting"] == "Hospital admission"].copy()
    # # print(len(hospital_tests), "non-covid PCR tests correspond to a hospital admission within 14 days")
    # print(len(matched_tests["StudyID"].unique())/ N_T, "of patients receiving non-covid PCR tests have a corresponding to a hospital admission within 14 days")
    # # print(len(hospital_tests["StudyID"].unique())/ N_T, "of non-covid PCR tests have a corresponding to a hospital admission within 14 days")


    # demographic_data = pd.read_sas('Data/Raw/KPSC/demographics.sas7bdat', encoding='utf-8')
    # # pivot by year
    # demographic_data = demographic_data.pivot_table(index='YEAR', values='n', aggfunc='sum').reset_index()
    # print(demographic_data)
    # N_S = demographic_data['n'].mean()
    # print(N_S)
    # for i, pathogen1 in enumerate(pathogens_of_interest):
    #     for pathogen2 in pathogens_of_interest[i+1:]:
    #         pt = pd.read_csv('Data/Processed/positivity_table_'+pathogen1+pathogen2+"date_restrictive.csv")
    #         prev1 = 0.05
    #         prev2 = 0.05
    #         # find number negative for both pathogens
    #         N_NA_NB = np.sum((pt[pathogen1] == 0) & (pt[pathogen2] == 0))
    #         N_AORB = np.sum((pt[pathogen1] == 1) | (pt[pathogen2] == 1))
    #         print(f"{pathogen1} and {pathogen2}: {N_NA_NB/(N_S*(1-prev1)*(1-prev2))}")
 
    # # start_time = time.time()
    # # df = process_data(test_data)
    # # print(f"Data processed in {time.time() - start_time} seconds")
    # # # save the processed data
    # # df.to_csv('Data/Processed/testing.csv', index=False)
    # # print(df["pathogen_group"].value_counts())
    # # load the processed data
    # df = pd.read_csv('Data/Processed/testing.csv')

    results = pd.read_csv('Data/Processed/viral_interference_CMH_tests.csv')
    results = results[(results['match_by_date'] == False) & (results['period'] == 'year_month')].copy()
    results_corrected = pd.read_csv('Data/Processed/viral_interference_CMH_tests_prevalence_correction.csv')
    results_corrected_reverse_order = pd.read_csv('Data/Processed/viral_interference_CMH_tests_prevalence_correction_reverse_order.csv')
    results_corrected_combined = pd.concat([results_corrected, results_corrected_reverse_order], ignore_index=True)
    fig, axes = plt.subplots(1,2,figsize=(13.3, 7.5))
    plot_heatmap(axes[0], results, pathogens_of_interest, title='Unadjusted')

    plot_heatmap(axes[1], results_corrected_combined, pathogens_of_interest, title='With prevalence adjustment',significance=0.005)
    plt.suptitle('Chochran-Mantel-Haenszel odds ratios for viral interference', fontsize=24)
    plt.tight_layout()
    plt.savefig('Figures/viral_interference_heatmap_combined.png', dpi=300)
