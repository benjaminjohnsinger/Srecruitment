import pandas as pd
# import jax.numpy as jnp
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

# # adjust plot text size
# plt.rcParams.update({'font.size':20})
# # text type is palatino
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Palatino']

plt.rcParams.update({'font.size':11})
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']


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
    elif "RHINOVIRUS" in pathogen:
        return "Rhinovirus"
    elif "ENTEROVIRUS" in pathogen:
        return "Enterovirus"
    else:
        return "Other"

def categorize_ndi(ndi):
    thresholds = jnp.array([-1,0,1])
    names = jnp.array(['Low', 'Medium-Low', 'Medium-High', 'High'])
    return names[jnp.digitize(ndi, thresholds)]

def categorize_age(age):
    thresholds = jnp.array([3, 12, 5*12, 18*12, 40*12, 65*12])
    names = jnp.array(['<3m', '3-11m', '1-4y', '5-17y', '18-39y', '40-64y', '>=65y'])
    return names[jnp.digitize(age, thresholds)]

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
def perform_cmh_test(positivity_table,pathogen1,pathogen2,period,prevalence_correction1=1,prevalence_correction2=1,threshold=0,Ps=None):
    # Create set of contingency tables for each age group and NDI group
    NAG = len(positivity_table['age_group'].unique())
    NNDIG = len(positivity_table['ndi_group'].unique())
    NP = len(positivity_table[period].unique())
    cmh_tables = jnp.zeros([NAG, NNDIG, NP, 2, 2])

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
                if contingency_table.shape == (2, 2) and jnp.sum(contingency_table) > threshold:
                    cmh_tables[i, j, k] = contingency_table
    
    # Collapse into age group and NDI group pairs
    cmh_tables = cmh_tables.reshape(-1, 2, 2)
    # Remove empty tables
    cmh_tables = cmh_tables[~jnp.all(cmh_tables == 0, axis=(1,2))]
    # # save to csv
    # cmh_tables_df = pd.DataFrame(cmh_tables.reshape(-1, 4), columns=['A', 'B', 'C', 'D'])
    # cmh_tables_df.to_csv('Data/Processed/cmh_tables.csv', index=False)
    
    # Calculate odds ratio
    D, C, B, A = cmh_tables[:, 0, 0], cmh_tables[:, 0, 1], cmh_tables[:, 1, 0], cmh_tables[:, 1, 1]
    print("Overall counts:",jnp.sum(A), jnp.sum(B), jnp.sum(C), jnp.sum(D))
    print("Prevalence:", jnp.sum(A + B) / jnp.sum(A + B + C + D), jnp.sum(A + C) / jnp.sum(A + B + C + D))
    if prevalence_correction1 != 1 or prevalence_correction2 != 1:
        ab = prevalence_correction1 / prevalence_correction2
        C = ab*C + (ab - 1)*A
        D = D + (1 - ab)*(A + C) + (prevalence_correction1 - 1)*(A + B + C + D)
        print("Corrected prevalence:", jnp.sum(A + B) / jnp.sum(A + B + C + D), jnp.sum(A + C) / jnp.sum(A + B + C + D))
    if Ps != None:
        pA, pB, pS = Ps
        A = A / max(pA, pB)
        B = B / pA
        C = C / pB
        D = D / pS
    T = A + B + C + D
    R = jnp.sum(A*D/T)/jnp.sum(B*C/T)

    # calculate test statistic
    N1, N2, M1, M2 = A + B, C + D, A + C, B + D
    chi_CMH = jnp.square(jnp.sum(A - N1*M1/T)) / jnp.sum(N1*N2*M1*M2/(T**2 * (T - 1)))

    # find p-value corresponding to the test statistic
    p_value = stats.chi2.sf(chi_CMH, 1)

    return R, p_value

def generate_tables(pathogens_of_interest,proportion_symptomatic=[],restrictive=False,period='year_month',load=False,threshold=0,bias=False):
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
            if bias:
                M, Ps = M_bias(pt, pathogen1, pathogen2, p_hospA=proportion_hospitalized[pathogen1],
                                    p_hospB=proportion_hospitalized[pathogen2])
                print(f"Bias factor: {M}")
            else:
                Ps = None
            R, p_value = perform_cmh_test(pt, pathogen1, pathogen2, period,
                prevalence_correction1=pc1, prevalence_correction2=pc2, threshold=threshold, Ps=Ps)
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
            results_df.to_csv('Data/Processed/viral_interference_CMH_tests2' + ["", "_prevalence_correction"][proportion_symptomatic != []] + ["", "_restrictive"][restrictive] + ["", "_Mbias"][bias] + "_threshold" + str(threshold) + '.csv', index=False)

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
        "HMPV": "hMPV",
        "Rhinovirus": "RV",}
    ax.set_xticklabels([pathogen_abbreviations[p.get_text()] for p in ax.get_xticklabels()], rotation=45, ha='right')
    ax.set_yticklabels([pathogen_abbreviations[p.get_text()] for p in ax.get_yticklabels()])

# approximation of bias based on conservative assumptions in Chin .. Lipsitch mBio 2024
def M_bias(positivity_table, pathogenA, pathogenB, hospA=0.03, p_hospA=0.85, hospB=0.03, p_hospB=0.85):
    # find the probability of appearing in study with neither pathogen
    N_NA_NB = jnp.sum((positivity_table[pathogenA] == 0) & (positivity_table[pathogenB] == 0))
    N_S = 4590046 # maximum Kaiser population
    pS = N_NA_NB / N_S # assuming very low prevalence
    # probability of appearing in study with each pathogen
    pA = hospA/p_hospA + pS
    pB = hospB/p_hospB + pS
    return pS/max(pA, pB), [pA, pB, pS] # return the bias factor and the probabilities of appearing in study with each pathogen and neither pathogen

pathogens_of_interest = ["Rhinovirus", "RSV", "HMPV", "Adenovirus", "Influenza", "Parainfluenza"]
proportion_symptomatic = {"All": 0.2894586894586895,
"Influenza": 0.5356125356125356,
"RSV": 0.341880341880342,
"Parainfluenza": 0.2279202279202281,
"HMPV": 0.6792022792022792,
"Rhinovirus": 0.2529914529914531,
"Adenovirus": 0.19601139601139594,
"Coronavirus": 0.2780626780626781} # Galanti et al. 2019 Epidemiology & Infection
proportion_hospitalized = {"Influenza": 0.6,
"RSV": 0.86,
"Parainfluenza": 0.85,
"HMPV": 0.85,
"Adenovirus": 0.85,
"Rhinovirus": 0.85} # Proportion recieving tests in data who are hospitalized

if __name__ == "__main__":
    from plotting import pathogen_names
    # for pathogen in pathogens_of_interest:
    #     print(pathogen, pathogen_names[pathogen])

    # test_data = pd.read_sas('Data/Raw/KPSC/testing.sas7bdat', encoding='utf-8')
    # test_data = test_data.loc[test_data["lab_type"]=="PCR"]
    # test_counts = pd.read_csv('Data/Processed/KPSC_test_counts.csv', index_col=0)
    # # bar chart
    # test_counts.plot(kind='bar', figsize=(10, 5), color="silver")
    # # log y
    # plt.yscale('log')
    # # 45 degree x labels
    # plt.xticks(rotation=45, ha='right')
    # # labelled dotted lines a y = 184200, 129355, and 54845
    # plt.plot([3, 14], [184200, 184200], color='#648FFF', linestyle='--', label='184200')
    # plt.plot([15, 17], [129355, 129355], color='#DC267F', linestyle='--', label='129355')
    # plt.plot([19, 23], [54845, 54845], color='#FFB000', linestyle='--', label='54845')
    # # add text to lines
    # plt.text(3, 184200, '184200', color='#648FFF', ha='left', va='bottom')
    # plt.text(15, 129355, '129355', color='#DC267F', ha='left', va='bottom')
    # plt.text(19, 54845, '54845', color='#FFB000', ha='left', va='bottom')
    # # replace end of x ticks with "..." after 20 characters
    # plt.gca().set_xticklabels([x.get_text()[:22] + '...' if len(x.get_text()) > 22 else x.get_text() for x in plt.gca().get_xticklabels()])
    # plt.tight_layout()
    # plt.ylabel('Number of tests')
    # plt.savefig('Figures/KPSC_test_counts.png', dpi=300)
    # # Number of tets of PARAINFLUENZA VIRUS 1, INFLUENZA VIRUS A, ENTEROVIRUS/RHINOVIRUS, HUMAN METAPNEUMOVIRUS VIRUS, and ADENOVIRUS
    # panel_pathogens = ["PARAINFLUENZA VIRUS 1","INFLUENZA VIRUS A","ENTEROVIRUS/RHINOVIRUS","HUMAN METAPNEUMOVIRUS VIRUS","ADENOVIRUS"]
    # print(test_data.loc[test_data["pathogen"].isin(panel_pathogens)].len(), "tests of panel pathogens")
    # print(test_data.loc[test_data["pathogen"].isin(panel_pathogens), "StudyID"].nunique(), "patients with panel pathogens")

    # print(len(test_data["StudyID"].unique()), "patients in the testing dataset")
    clinical_data = pd.read_sas('Data/Raw/KPSC/clinical_20241202.sas7bdat', encoding='utf-8')
    # print(len(clinical_data["StudyID"].unique()), "patients in the clinical dataset")
    clinical_data["Date"] = pd.to_datetime(clinical_data["YEAR"].astype(int).astype(str) + '-10-01') + pd.to_timedelta(clinical_data["dx_days"],unit='D')
    print(clinical_data["Date"].min(), clinical_data["Date"].max())
    print(("00002734" in clinical_data["StudyID"].astype(str).values))

    # for pathogen in pathogens_of_interest:
    #     print(pathogen_names[pathogen])
    #     test_data_pathogen = test_data[test_data['pathogen'].isin(pathogen_names[pathogen])].copy()
    #     N_T = len(test_data_pathogen["StudyID"].unique())
    #     print(N_T, f"patients with {pathogen} PCR tests")
    #     clinical_tests = test_data_pathogen[test_data_pathogen["StudyID"].isin(clinical_data["StudyID"])]
    #     clinical_data_pathogen = clinical_data[clinical_data["StudyID"].isin(clinical_tests["StudyID"])]
    #     print(clinical_data_pathogen["setting"].value_counts())
    #     print(len(clinical_data_pathogen["StudyID"].unique()), f"hospitalized patients in the clinical dataset with {pathogen} PCR tests")
    #     clinical_tests["Date"] = pd.to_datetime(clinical_tests["YEAR"].astype(int).astype(str) + '-10-01') + pd.to_timedelta(clinical_tests["lab_days"],unit='D')
    #     clinical_tests = clinical_data_pathogen.merge(clinical_tests[["StudyID","Date"]],on="StudyID",how="left")
    #     clinical_tests = clinical_tests.rename(columns={"Date_x":"Clinical date","Date_y":"Test date"})
    #     matched_tests = clinical_tests[jnp.abs((pd.to_datetime(clinical_tests["Clinical date"]) - pd.to_datetime(clinical_tests["Test date"])).dt.days) <= 14]
    #     print(len(matched_tests), f" {pathogen} PCR tests correspond to a hospital admission within 14 days")
    #     # hospital_tests = matched_tests[matched_tests["setting"] == "Hospital admission"].copy()
    #     # print(len(hospital_tests), f"non-covid {pathogen} PCR tests correspond to a hospital admission within 14 days")
    #     print(len(matched_tests["StudyID"].unique())/ N_T, f"of patients receiving {pathogen} PCR tests have a corresponding to a hospital admission within 14 days")
    #     # print(len(hospital_tests["StudyID"].unique())/ N_T, f"of non-covid {pathogen} PCR tests have a corresponding to a hospital admission within 14 days")


    # demographic_data = pd.read_sas('Data/Raw/KPSC/demographics.sas7bdat', encoding='utf-8')
    # # pivot by year
    # demographic_data = demographic_data.pivot_table(index='YEAR', values='n', aggfunc='sum').reset_index()
    # print(demographic_data)
    # N_S = demographic_data['n'].mean()
    # print(N_S)
    # for i, pathogen1 in enumerate(pathogens_of_interest):
    #     for pathogen2 in pathogens_of_interest[i+1:]:
    #         pt = pd.read_csv('Data/Processed/positivity_table_'+pathogen1+pathogen2+"date_restrictive.csv")
    #         print(M_bias(pt, pathogen1, pathogen2, p_hospA=proportion_hospitalized[pathogen1], p_hospB=proportion_hospitalized[pathogen2]))
    #         prev1 = 0.1
    #         prev2 = 0.1
    #         # find number negative for both pathogens
    #         N_NA_NB = jnp.sum((pt[pathogen1] == 0) & (pt[pathogen2] == 0))
    #         N_AORB = jnp.sum((pt[pathogen1] == 1) | (pt[pathogen2] == 1))
    #         print(f"Proportion of patients in study for {pathogen1} and {pathogen2}: {(N_NA_NB+N_AORB)/N_S}")
    #         print(f"Probability of appearing in study given negativity for both, assuming very low prevalence: {N_NA_NB/N_S}")
    #         print(f"Probability of appearing in study given negativity for both, assuming high prevalence: {N_NA_NB/(N_S*(1-prev1)*(1-prev2))}")
    #         print("\n")
 
    # start_time = time.time()
    # df = process_data(test_data)
    # print(f"Data processed in {time.time() - start_time} seconds")
    # # save the processed data
    # df.to_csv('Data/Processed/testing.csv', index=False)
    # print(df["pathogen_group"].value_counts())
    # load the processed data
    # df = pd.read_csv('Data/Processed/testing.csv')
    # generate_tables(pathogens_of_interest, restrictive=True, period='year_month', load=True, threshold=0, bias=True)

    # results = pd.read_csv('Data/Processed/viral_interference_CMH_tests.csv')
    # results = results[(results['match_by_date'] == True) & (results['period'] == 'year_month')].copy()
    # results_corrected = pd.read_csv('Data/Processed/viral_interference_CMH_tests_restrictive_Mbias_threshold0.csv')
    # # results_corrected_reverse_order = pd.read_csv('Data/Processed/viral_interference_CMH_tests_prevalence_correction_reverse_order.csv')
    # # results_corrected_combined = pd.concat([results_corrected, results_corrected_reverse_order], ignore_index=True)
    # fig, axes = plt.subplots(1,2,figsize=(6.5, 3.5))
    # # Create the heatmaps without color bars
    # plot_heatmap(axes[0], results, pathogens_of_interest, title='Naïve', significance=None)
    # plot_heatmap(axes[1], results_corrected, pathogens_of_interest, title='With correction factor', significance=None)
    # plt.tight_layout()
    # # remove color bar
    # axes[0].collections[0].colorbar.remove()
    # axes[1].collections[0].colorbar.remove()

    # # Add a shared color bar
    # cbar = fig.colorbar(axes[0].collections[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    # cbar.set_label('Odds Ratio')
    # # plt.suptitle('Chochran-Mantel-Haenszel odds ratios for viral interference')

    # plt.savefig('Figures/viral_interference_heatmap_wMbias2.png', dpi=300)
