import pandas as pd
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from sas7bdat import SAS7BDAT
from matplotlib import cm as colormaps
from Parameters.census_population import *
import pickle
import time
import sys
from matplotlib.patches import Rectangle
from Parameters.census_population import AGE_GROUPS, AGE_GROUP_NAMES

################ Data processing functions ################

##### function to calculate proportion positive tests for a given pathogen in a moving window, and multiply by population-proportional incidence of ARI hospitalizations ######
# daily_hospitalization_rates = pd.read_csv('Data/Processed/KPSC_ARI_hospitalization_rates_by_day_age_group.csv',index_col=0,parse_dates=True)

def calculate_proportion_positive_incidence(pathogen, window_size=28, weighting_factor=0.1, aggregation='D', sum_age_groups=False, save_counts=False, pp_only=False, hosp=False, salvage=True):
    pop_by_age_group_month = pd.read_csv('Data/Processed/KPSC_population_by_age_group_monthly.csv', index_col=0, parse_dates=['month_start'])
    daily_hospitalization_counts = pd.read_csv('Data/Processed/KPSC_ARI_nonCOVID_hospitalizations_by_day_age_group.csv', index_col=0, parse_dates=True)
    daily_test_counts_complete = pd.read_csv('Data/Processed/KPSC_ARI_hospitalized_pathogen'+['','_unsalvage'][not salvage]+'_panel_test_counts_by'+['','_hosp'][hosp]+'_day_pathogen_age_group.csv',index_col=0,parse_dates=True)

    # Filter for specified pathogens
    pathogen_data = daily_test_counts_complete[daily_test_counts_complete['pathogen']==pathogen]
    
    # Pivot to get positive and total counts by date and age group
    date_name = "Hospitalization date" if hosp else "Test date"
    positive_counts = pathogen_data[pathogen_data['result_type'] == 'Positive'].pivot_table(
        index=date_name, columns='age_group', values='count', aggfunc='sum', fill_value=0)
    total_counts = pathogen_data[pathogen_data['result_type'] == 'Total'].pivot_table(
        index=date_name, columns='age_group', values='count', aggfunc='sum', fill_value=0)
    
    # Reorder columns to match AGE_GROUP_NAMES
    positive_counts = positive_counts.reindex(columns=AGE_GROUP_NAMES, fill_value=0)
    total_counts = total_counts.reindex(columns=AGE_GROUP_NAMES, fill_value=0)
    
    # save a three layered array with total counts, positive counts, and total hospitalizations for each date and age group
    if save_counts:
        # Fill missing values with zeros
        positive_counts_filled = positive_counts.fillna(0)
        total_counts_filled = total_counts.fillna(0)
        # Create complete date range covering all test dates
        date_range = pd.date_range(start=min(positive_counts.index.min(), total_counts.index.min()), 
                      end=max(positive_counts.index.max(), total_counts.index.max()), 
                      freq='D')
        
        # Reindex to include all dates and fill with zeros
        positive_counts_filled = positive_counts.reindex(date_range, fill_value=0)
        total_counts_filled = total_counts.reindex(date_range, fill_value=0)
        daily_hospitalization_counts_filled = daily_hospitalization_counts.reindex(date_range, fill_value=0)
        # Save as separate CSV files with no index or column names
        positive_counts_filled.to_csv(f'Data/Processed/KPSC_panel_{pathogen}_positive_counts'+['', '_hospday'][hosp]+'.csv', header=False, index=False)
        total_counts_filled.to_csv(f'Data/Processed/KPSC_panel_{pathogen}_total_counts'+['', '_hospday'][hosp]+'.csv', header=False, index=False)
        daily_hospitalization_counts_filled.to_csv(f'Data/Processed/KPSC_panel_hospitalizations_noCOVID.csv', header=False, index=False)

    if aggregation == "D":
        # When both positive and total counts are zero, set proportion to 0
        prop_pos = positive_counts / total_counts.replace(0, np.nan)
        prop_pos = prop_pos.fillna(0)  # Fill NaN values (from 0/0) with 0

        # Create exponential weights centered on middle of window
        half_window = window_size // 2
        weights = np.exp(-weighting_factor * np.abs(np.arange(window_size) - half_window))
        weights = weights / weights.sum()  # Normalize weights to sum to 1
        
        # Apply weighted rolling window
        prop_pos_smoothed = pd.DataFrame(index=prop_pos.index, columns=prop_pos.columns)
        for col in prop_pos.columns:
            prop_pos_smoothed[col] = prop_pos[col].rolling(
                window=window_size, center=True, min_periods=1
            ).apply(lambda x: np.average(x, weights=weights[:len(x)]) if len(x) > 0 else 0, raw=False)
        
        if pp_only:
            if sum_age_groups:
                # Sum positive and total counts across age groups first
                positive_counts_summed = positive_counts.sum(axis=1)
                total_counts_summed = total_counts.sum(axis=1)
                prop_pos_summed = positive_counts_summed / total_counts_summed.replace(0, np.nan)
                prop_pos_summed = prop_pos_summed.fillna(0)
                
                # Apply smoothing to the summed proportion
                prop_pos_smoothed_summed = prop_pos_summed.rolling(
                    window=window_size, center=True, min_periods=1
                ).apply(lambda x: np.average(x, weights=weights[:len(x)]) if len(x) > 0 else 0, raw=False)
                
                return prop_pos_smoothed_summed.to_frame(name='Total')
            else:
                return prop_pos_smoothed

        # Align with hospitalization rates and calculate incidence
        pop_by_age_group_daily = pop_by_age_group_month.resample('D').ffill()
        pop_by_age_group_daily = pop_by_age_group_daily.reindex(columns=AGE_GROUP_NAMES, fill_value=0)
        daily_hospitalization_rates = daily_hospitalization_counts.div(pop_by_age_group_daily, axis=1)
        aligned_hosp_rates = daily_hospitalization_rates.reindex(prop_pos_smoothed.index)
        incidence = prop_pos_smoothed * aligned_hosp_rates
    else:
        # Resample to desired aggregation
        positive_counts_agg = positive_counts.resample(aggregation).sum()
        total_counts_agg = total_counts.resample(aggregation).sum()
        
        # Calculate proportion positive
        prop_pos_agg = positive_counts_agg / total_counts_agg.replace(0, np.nan)
        prop_pos_agg = prop_pos_agg.fillna(0)  # Fill NaN values (from 0/0) with 0
        
        if pp_only:
            if sum_age_groups:
                # Sum positive and total counts across age groups
                positive_counts_summed = positive_counts_agg.sum(axis=1)
                total_counts_summed = total_counts_agg.sum(axis=1)
                prop_pos_summed = positive_counts_summed / total_counts_summed.replace(0, np.nan)
                prop_pos_summed = prop_pos_summed.fillna(0)
                
                return prop_pos_summed.to_frame(name='Total')
            else:
                return prop_pos_agg
                
        # Calculate incidence
        pop_by_age_group_agg = pop_by_age_group_month.resample(aggregation).ffill()
        pop_by_age_group_agg = pop_by_age_group_agg.reindex(columns=AGE_GROUP_NAMES, fill_value=0)
        daily_hospitalization_counts_agg = daily_hospitalization_counts.resample(aggregation).sum()
        daily_hospitalization_rates_agg = daily_hospitalization_counts_agg.div(pop_by_age_group_agg, axis=1)
        aligned_hosp_rates_agg = daily_hospitalization_rates_agg.reindex(prop_pos_agg.index)
        incidence = prop_pos_agg * aligned_hosp_rates_agg
    
    # Sum age groups at the end if requested
    if sum_age_groups:
        # Weight by population when summing
        if aggregation == "D":
            pop_weights = pop_by_age_group_daily.reindex(incidence.index)
        else:
            pop_weights = pop_by_age_group_agg.reindex(incidence.index)
        
        # Calculate weighted sum
        weighted_incidence = (incidence * pop_weights).sum(axis=1)
        total_population = pop_weights.sum(axis=1)
        incidence = (weighted_incidence / total_population).to_frame(name='Total')
    
    return incidence

def load_and_filter_test_data():
    # time_start = time.time()
    test_data1 = pd.read_sas('Data/Raw/KPSC/testing.sas7bdat', format='sas7bdat', encoding='utf-8')
    test_data2 = pd.read_sas('Data/Raw/KPSC/testing_20250818.sas7bdat', format='sas7bdat', encoding='utf-8')
    test_data = pd.concat([test_data1, test_data2], ignore_index=True)
    
    # load RSV salvage test data from Data/Raw/KPSC/Viral_Transmission_RSV_TestResults_20250623.xlsx
    rsv_salvage_data = pd.read_excel('Data/Raw/KPSC/Viral_Transmission_RSV_TestResults_20250623.xlsx')
    
    # Rename columns to match pathogen names in test_data
    rsv_salvage_data = rsv_salvage_data.rename(columns={
        "adenovirus": "ADENOVIRUS",
        "chlamydia": "CHLAMYDOPHILA PNEUMONIAE",
        "coronavirus": "CORONAVIRUS",
        "covid": "SARS-COV-2 (COVID-19)",
        "enterovirus": "ENTEROVIRUS/RHINOVIRUS",
        "fluA": "INFLUENZA A",
        "fluA_H1": "INFLUENZA A H1N1 2009",
        "fluA_H3": "INFLUENZA A VIRUS SUBTYPE H3",
        "fluA_h1n1": "INFLUENZA A VIRUS SUBTYPE H1",
        "fluB": "INFLUENZA B",
        "hmpv": "HUMAN METAPNEUMOVIRUS VIRUS",
        "mycoplasma": "MYCOPLASMA PNEUMONIAE",
        "paraflu1": "PARAINFLUENZA VIRUS 1",
        "paraflu2": "PARAINFLUENZA VIRUS 2",
        "paraflu3": "PARAINFLUENZA VIRUS 3",
        "paraflu4": "PARAINFLUENZA VIRUS 4",
        "rsvA": "RESPIRATORY SYNCYTIAL VIRUS SUBTYPE A",
        "rsvB": "RESPIRATORY SYNCYTIAL VIRUS SUBTYPE B"
    })
    
    # Reshape rsv_salvage_data to long format
    rsv_salvage_data = rsv_salvage_data.melt(
        id_vars=["StudyID", "collect_date", "lab_type"],
        var_name="pathogen",
        value_name="result_val"
    )
    
    # Create date columns for efficient matching
    test_data.loc[:, 'Test date'] = (
        pd.to_datetime(test_data["YEAR"].astype(int).astype(str) + '-10-01') +
        pd.to_timedelta(test_data["lab_days"].astype(int), unit='D')
    ).dt.normalize()
    
    rsv_salvage_data.loc[:, 'collect_date'] = pd.to_datetime(
        rsv_salvage_data['collect_date']
    ).dt.normalize()
    
    # Prepare columns for merge
    merge_cols = ['StudyID', 'Test date', 'pathogen', 'result_val', 'lab_type']
    rsv_salvage_renamed = rsv_salvage_data.rename(columns={'collect_date': 'Test date'})
    
    # Find rows that don't match rsv_salvage_data
    merged = test_data.merge(
        rsv_salvage_renamed[merge_cols],
        on=merge_cols,
        how='left',
        indicator=True
    )
    
    # Keep only rows not in rsv_salvage_data
    test_data = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge']).copy()
    
    # print(f"Time to filter test data: {time.time() - time_start:.2f}s")
    return test_data

def filter_to_panel_tests(test_data, date_name="Test date"):
    pathogen_names = {
    "RSV": ["RESPIRATORY SYNCYTIAL VIRUS","RESPIRATORY SYNCYTIAL VIRUS SUBTYPE A","RESPIRATORY SYNCYTIAL VIRUS SUBTYPE B",],
    "InfluenzaA": ["INFLUENZA A","INFLUENZA A H1N1 2009","INFLUENZA A VIRUS","INFLUENZA A VIRUS SUBTYPE H1","INFLUENZA A VIRUS SUBTYPE/HEMAGGLUTININ H3","INFLUENZA VIRUS A",],
    "InfluenzaB": ["INFLUENZA B","INFLUENZA VIRUS B",],
    "Metapneumovirus": ["HUMAN METAPNEUMOVIRUS VIRUS",],
    "Adenovirus": ["ADENOVIRUS",],
    "Parainfluenza1": ["PARAINFLUENZA VIRUS 1"],
    "Parainfluenza2": ["PARAINFLUENZA VIRUS 2"],
    "Parainfluenza3": ["PARAINFLUENZA VIRUS 3"],
    "Parainfluenza4": ["PARAINFLUENZA VIRUS 4"],
    "Rhinovirus": ["ENTEROVIRUS/RHINOVIRUS"],
    "Pertussis": ["BORDETELLA PERTUSSIS"],
    "M.pneumoniae": ["MYCOPLASMA PNEUMONIAE"],
    "C.pneumoniae": ["CHLAMYDOPHILA PNEUMONIAE"],
    "SARS-CoV-2": ["SARS-COV-2 (COVID-19)"],
    "Enterovirus": ["ENTEROVIRUS/RHINOVIRUS"],
    }
    # replace pathogen names with group names
    # Create a single mapping dictionary from all pathogen names to group names
    reverse_names = [{v:k for v in values} for k,values in pathogen_names.items()]
    reverse_names =  {k:v for d in reverse_names for k,v in d.items()}
    test_data.loc[:,"pathogen"] = test_data["pathogen"].map(reverse_names)
    test_data = test_data[test_data["pathogen"].notna()]

    pathogen_list = ["InfluenzaA","InfluenzaB","RSV","Metapneumovirus","Adenovirus","Parainfluenza3"]
    # Keep only panel tests, i.e. there is a test for each pathogen in the list (that is, containing the string for each pathogen in the list in the pathogen column) for a given StudyID and test date
    panel_test_groups = test_data.groupby(["StudyID", date_name])["pathogen"].apply(lambda x: all(any(p == pathogen for pathogen in x) for p in pathogen_list))
    test_data = test_data.set_index(["StudyID", date_name]).loc[panel_test_groups[panel_test_groups].index].reset_index()
    return test_data

def load_and_filter_hospitalization_data(exclude_covid=True):
    # time_start = time.time()
    clinical_data1 = pd.read_sas('Data/Raw/KPSC/clinical_20241202.sas7bdat', format='sas7bdat', encoding='utf-8')
    clinical_data2 = pd.read_sas('Data/Raw/KPSC/clinical_20260203.sas7bdat', format='sas7bdat', encoding='utf-8')
    clinical_data = pd.concat([clinical_data1, clinical_data2], ignore_index=True)
    # print("Time to load clinical data: ",time.time()-time_start)

    clinical_data = clinical_data[clinical_data["dxgroup"] == "ARI"]

    clinical_data = clinical_data[clinical_data["setting"] == "Hospital admission"]

    clinical_data.loc[:, "Hospitalization date"] = (
        pd.to_datetime(clinical_data["YEAR"].astype(int).astype(str) + '-10-01') +
        pd.to_timedelta(clinical_data["dx_days"].astype(int), unit='D')
    ).dt.normalize()

    if exclude_covid:
        covid_records = clinical_data[clinical_data["CODE"] == "U07.1"].copy()
        # print(covid_records.head())
        # what is the earliest date in covid_records
        print("Earliest COVID record date: ", covid_records["Hospitalization date"].min())

        if not covid_records.empty:
            # Merge clinical data with COVID records to find matches within 14 days
            merged = clinical_data.reset_index(drop=True).merge(
                covid_records[["StudyID", "Hospitalization date"]].rename(columns={"Hospitalization date": "COVID_date"}),
                on="StudyID",
                how="left"
            )
            
            # Calculate days difference
            merged["days_diff"] = np.abs((pd.to_datetime(merged["Hospitalization date"]) - pd.to_datetime(merged["COVID_date"])).dt.days)
            
            # Mark records with COVID within 14 days
            merged["exclude"] = merged["days_diff"] <= 14
            
            # Get indices to exclude (keep only the first match per record)
            exclude_mask = merged["exclude"].fillna(False)
            
            print(f"Excluding {exclude_mask.sum()} records with COVID diagnoses within 14 days")
            
            # Keep only rows where exclude is False
            clinical_data = merged[~exclude_mask].drop(columns=["COVID_date", "days_diff", "exclude"]).reset_index(drop=True)

    return clinical_data

def bin_age_groups(data):
    from Parameters.census_population import AGE_GROUPS, AGE_GROUP_NAMES
    bins = [group[0] for group in AGE_GROUPS] + [AGE_GROUPS[-1][-1] + 1]
    data.loc[:,"AGE_GROUP"] = pd.cut(data["age_in_mo"],bins=bins,labels=AGE_GROUP_NAMES,right=False)
    return data

def merge_positive_tests(test_data, clinical_data, clinical_date_name="Hospitalization date", test_date_name="Test date"):
    test_data = test_data[test_data["StudyID"].isin(clinical_data["StudyID"])].copy()
    positive_tests = test_data[test_data["result_val"] == 'Positive'].copy()
    # # find date of from clinical_data for each study ID and match to positive tests
    positive_tests = clinical_data.merge(positive_tests[["StudyID",test_date_name,"pathogen","lab_type","lab_days"]],on="StudyID",how="left")
    # # # keep only rows where Hospitalization date is within 14 days of Test date
    positive_tests = positive_tests[np.abs((pd.to_datetime(positive_tests[clinical_date_name]) - pd.to_datetime(positive_tests[test_date_name])).dt.days) <= 14]
    # keep only one test per pathogen and hospitalization
    positive_tests = positive_tests.sort_values(by=["StudyID","pathogen",test_date_name], ascending=[True,True,True])
    # find groups of tests within 14 days of each other with the same StudyID and pathogen
    positive_tests.loc[:,"diff"] = positive_tests.groupby(["StudyID","pathogen"])[test_date_name].diff().dt.days
    # for groups of tests where diff is less than 14 days, keep only the first test
    positive_tests = positive_tests[(positive_tests["diff"].isna()) | (positive_tests["diff"] > 14)]
    positive_tests = positive_tests.drop(columns=["diff"])
    return positive_tests



# if __name__ == "__main__":
#     # test function
#     # incidence = calculate_proportion_positive_incidence(["INFLUENZA B","INFLUENZA VIRUS B","INFLUENZA VIRUS A+B"], aggregation="ME", window_size=28, weighting_factor=np.log(2))
#     # plot
#     from plotting import hsv_colors
#     pathogen="InfluenzaA"
#     for pi,pathogen in enumerate(["InfluenzaA","RSV","Adenovirus","InfluenzaB","Metapneumovirus","Parainfluenza3"]):
#         fig, ax = plt.subplots(4,2,figsize=(13.3,7.5),sharex=True)
#         incidence = calculate_proportion_positive_incidence(pathogen, aggregation="D", window_size=1, weighting_factor=0, sum_age_groups=False, save_counts=True, pp_only=False, hosp=True)
#         # incidence.index = incidence.index.to_period('M').to_timestamp() + pd.offsets.Day(14) # shift to middle of month
#         incidence.to_csv(f'Data/Processed/KPSC_panel_proportion_positive_ARI_nonCOVID_{pathogen}_incidence_age_hosp_daily.csv')
#         for i,age_group in enumerate(AGE_GROUP_NAMES):
#             # incidence[age_group].plot(ax=ax[pi//2, pi%2],color=hsv_colors[i],label=age_group)/
#             ax[i%4, i//4].plot(incidence.index, incidence[age_group] * 100000, color="k", label=age_group)
#             ax[i%4, i//4].set_title(f"{age_group}")
#         # (incidence['Total'] * 10000).plot(ax=ax[pi//2, pi%2],color='k',label='Total')
#         # ax[pi//3, pi%3].plot(incidence.index, incidence['Total'] * 10000, color='k', label='Total')
#         # ax[i//2, i%2].set_title(f"{pathogen}")
#         ax[1, 0].set_ylabel('Incidence per 100k')
#         # ax[3,0].set_xticks(incidence.index[::52], [str(year) for year in incidence.index.year[::52]], rotation=45)
#         # remove last axis
#         ax[3,1].axis('off')
#         # ax[2,0].set_xlabel('Date')
#         # ax[2,1].set_xlabel('Date')
#         # ax[0,1].legend(title="Age group", loc = "upper right", ncol=2)
#         # only include every other x tick and label with year at 45 degree angle
#         plt.tight_layout()
#         # plt.savefig(f"Figures/{pathogen}_age_group_panels_hospitalization_incidence_monthly_withsalvage.png",dpi=300)
#     # plt.savefig("Figures/KPSC_panel_tests_by_age_group_monthly.png",dpi=300)
#     # plt.savefig("Figures/KPSC_unsalvage_panel_proportion_positive_incidence_pathogen_age_hosp_daily_slide.png",dpi=300)


#### old incidence data (new code)
if __name__ == "__main__":
    time_start = time.time()
    test_data = load_and_filter_test_data()
    time_test = time.time()
    print(f"Time to filter test data: {time_test - time_start:.2f}s")

    test_data = filter_to_panel_tests(test_data)
    time_panel = time.time()
    print(f"Time to filter panel tests: {time_panel - time_test:.2f}s")

    hospitalization_data = load_and_filter_hospitalization_data(exclude_covid=True)
    time_hosp = time.time()
    print(f"Time to filter hospitalization data: {time_hosp - time_test:.2f}s")

    hospitalization_data = bin_age_groups(hospitalization_data)

    combined_data = merge_positive_tests(test_data, hospitalization_data)
    time_merge = time.time()
    print(f"Time to merge positive tests with hospitalization data: {time_merge - time_hosp:.2f}s")

    combined_data.to_csv('Data/Processed/KPSC_unsalvage_panel_positive_matched_noncovid_ARI_hospitalizations.csv',index=False)

    combined_data = pd.read_csv('Data/Processed/KPSC_unsalvage_panel_positive_matched_noncovid_ARI_hospitalizations.csv', parse_dates=["Hospitalization date"])
    # for each pathogen, aggregate the number of positive tests by age group and day
    for pathogen in ["InfluenzaA","InfluenzaB","RSV","Metapneumovirus","Adenovirus","Parainfluenza3"]:
        pathogen_data = combined_data[combined_data["pathogen"].str.contains(pathogen,na=False)].copy()
        pathogen_data = pathogen_data.groupby(["Hospitalization date","AGE_GROUP"], observed=True)["StudyID"].count().reset_index()
        pathogen_data.rename(columns={"StudyID":"Positive tests"},inplace=True)
        # wide format with age groups as columns
        pathogen_data = pathogen_data.pivot(index="Hospitalization date", columns="AGE_GROUP", values="Positive tests").fillna(0).reset_index()
        # set order of age group columns
        pathogen_data = pathogen_data[["Hospitalization date"] + AGE_GROUP_NAMES]
        # fill dates from 2015-10-01 to 2025-05-01 with 0 positive tests for each age group
        all_dates = pd.date_range(start="2015-10-01", end="2025-05-01")
        pathogen_data = pathogen_data.set_index("Hospitalization date").reindex(all_dates).fillna(0).rename_axis("Hospitalization date").reset_index()
        pathogen_data.to_csv(f'Data/Processed/KPSC_unsalvage_panel_positive_{pathogen}_matched_noncovid_ARI_hospitalizations.csv',index=False)
        pop_by_age_group_month = pd.read_csv('Data/Processed/KPSC_population_by_age_group_monthly.csv', index_col=0, parse_dates=['month_start'])
        pop_by_age_group_month = pop_by_age_group_month.reindex(columns=AGE_GROUP_NAMES)
        pop_by_age_group_daily = pop_by_age_group_month.resample('D').ffill()
        # get proportional incidence by dividing positive tests by population
        pathogen_data[AGE_GROUP_NAMES] = pathogen_data[AGE_GROUP_NAMES].div(
            pop_by_age_group_daily.reindex(pathogen_data["Hospitalization date"]).values, axis=0
        )
        pathogen_data.to_csv(f'Data/Processed/KPSC_unsalvage_panel_positive_{pathogen}_matched_noncovid_ARI_hospitalizations_proportional_incidence.csv',index=False)
    


# ############### CDC data ###############
# ### full NREVSS data
# data = pd.read_excel('Data/Raw/NREVSS_all.xlsx',sheet_name='Final')
# # drop REGNAME column
# data.drop(columns=["REGNAME"],inplace=True)

# # sum data (RSVpos,RSVtest,PIV3pos,PIVtest,RAdenopos,RAdenotest,HMetapneumopos,HMetapneumotest) with same date, region, and test type
# data = data.groupby(["RepWeekDate","HHS_REGION","TestType"]).sum().reset_index()
# data["Date"] = pd.to_datetime(data["RepWeekDate"],format='%m/%d/%Y')

# data["RSV Percent Positive"] = 100*data["RSVpos"]/data["RSVtest"]
# data["PIV3 Percent Positive"] = 100*data["PIV3pos"]/data["PIVtest"]
# data["Adenovirus Percent Positive"] = 100*data["RAdenopos"]/data["RAdenotest"]
# data["Metapneumovirus Percent Positive"] = 100*data["HMetapneumopos"]/data["HMetapneumotest"]

# # add "Region" in front of "HHS_REGION" column
# data["Region"] = "Region " + data["HHS_REGION"].astype(str)

# # index and pivot
# antigen_pp_RSV = data.loc[data["TestType"] == 1].pivot(index="Date",columns="Region",values="RSV Percent Positive")
# antigen_pp_PIV3 = data.loc[data["TestType"] == 1].pivot(index="Date",columns="Region",values="PIV3 Percent Positive")
# antigen_pp_AdV = data.loc[data["TestType"] == 1].pivot(index="Date",columns="Region",values="Adenovirus Percent Positive")
# antigen_pp_MPV = data.loc[data["TestType"] == 1].pivot(index="Date",columns="Region",values="Metapneumovirus Percent Positive")

# culture_pp_RSV = data.loc[data["TestType"] == 2].pivot(index="Date",columns="Region",values="RSV Percent Positive")
# culture_pp_PIV3 = data.loc[data["TestType"] == 2].pivot(index="Date",columns="Region",values="PIV3 Percent Positive")
# culture_pp_AdV = data.loc[data["TestType"] == 2].pivot(index="Date",columns="Region",values="Adenovirus Percent Positive")
# culture_pp_MPV = data.loc[data["TestType"] == 2].pivot(index="Date",columns="Region",values="Metapneumovirus Percent Positive")

# pcr_pp_RSV = data.loc[data["TestType"] == 4].pivot(index="Date",columns="Region",values="RSV Percent Positive")
# pcr_pp_PIV3 = data.loc[data["TestType"] == 4].pivot(index="Date",columns="Region",values="PIV3 Percent Positive")
# pcr_pp_AdV = data.loc[data["TestType"] == 4].pivot(index="Date",columns="Region",values="Adenovirus Percent Positive")
# pcr_pp_MPV = data.loc[data["TestType"] == 4].pivot(index="Date",columns="Region",values="Metapneumovirus Percent Positive")

# print(pcr_pp_RSV)

# # save to csv
# antigen_pp_RSV.to_csv('Data/Processed/NREVSS_Antigen_PercentPositive_RSV.csv')
# antigen_pp_PIV3.to_csv('Data/Processed/NREVSS_Antigen_PercentPositive_PIV3.csv')
# antigen_pp_AdV.to_csv('Data/Processed/NREVSS_Antigen_PercentPositive_AdV.csv')
# antigen_pp_MPV.to_csv('Data/Processed/NREVSS_Antigen_PercentPositive_MPV.csv')
# culture_pp_RSV.to_csv('Data/Processed/NREVSS_Culture_PercentPositive_RSV.csv')
# culture_pp_PIV3.to_csv('Data/Processed/NREVSS_Culture_PercentPositive_PIV3.csv')
# culture_pp_AdV.to_csv('Data/Processed/NREVSS_Culture_PercentPositive_AdV.csv')
# culture_pp_MPV.to_csv('Data/Processed/NREVSS_Culture_PercentPositive_MPV.csv')
# pcr_pp_RSV.to_csv('Data/Processed/NREVSS_PCR_PercentPositive_RSV.csv')
# pcr_pp_PIV3.to_csv('Data/Processed/NREVSS_PCR_PercentPositive_PIV3.csv')
# pcr_pp_AdV.to_csv('Data/Processed/NREVSS_PCR_PercentPositive_AdV.csv')
# pcr_pp_MPV.to_csv('Data/Processed/NREVSS_PCR_PercentPositive_MPV.csv')

# # plot
# # antigen_pp_RSV = antigen_pp_RSV[["Region " + str(i) for i in range(1,9)]+["Region 10"]+["Region 9"]]
# # antigen_pp_PIV3 = antigen_pp_PIV3[["Region " + str(i) for i in range(1,9)]+["Region 10"]+["Region 9"]]
# # antigen_pp_AdV = antigen_pp_AdV[["Region " + str(i) for i in range(1,9)]+["Region 10"]+["Region 9"]]
# # antigen_pp_MPV = antigen_pp_MPV[["Region " + str(i) for i in range(1,9)]+["Region 10"]+["Region 9"]]
# # colors = colormaps.get_cmap('Greys',9)(jnp.linspace(1,0.3,9)).tolist()
# # colors.append('red')
# # fig, ax = plt.subplots(4,1,figsize=(13.3,10),sharex=True,sharey=True)
# # antigen_pp_RSV.plot(ax=ax[0],legend=False,color=colors)
# # antigen_pp_PIV3.plot(ax=ax[1],legend=False,color=colors)
# # antigen_pp_AdV.plot(ax=ax[2],legend=False,color=colors)
# # antigen_pp_MPV.plot(ax=ax[3],legend=False,color=colors)
# # ax[0].set_ylabel("RSV")
# # ax[1].set_ylabel("PIV3")
# # ax[2].set_ylabel("Adenovirus")
# # ax[3].set_ylabel("Metapneumovirus")
# # ax[3].set_xlabel("Date")
# # fig.suptitle("NREVSS Percent Antigen Positive by HHS Region")
# # plt.tight_layout()
# # plt.savefig('Figures/NREVSS_Antigen_PercentPositive.png')

# ### RSV data
# rsv_pre2020 = pd.read_csv('Data/Raw/Respiratory_Syncytial_Virus_Laboratory_Data__NREVSS_.csv')
# rsv_post2020 = pd.read_csv('Data/Raw/Percent_Positivity_of_Respiratory_Syncytial_Virus_Nucleic_Acid_Amplification_Tests_by_HHS_Region__National_Respiratory_and_Enteric_Virus_Surveillance_System_20250213.csv')

# # get dates from column "Week ending Date" which are in format e.g. 22JUL2017
# rsv_pre2020["Date"] = pd.to_datetime(rsv_pre2020["Week ending Date"],format='%d%b%Y')
# # get dates from the column "mmwrweek_end" which are in fomrat e.g. 04/11/2020 12:00:00 AM
# rsv_post2020["Date"] = pd.to_datetime(rsv_post2020["mmwrweek_end"],format='%m/%d/%Y %I:%M:%S %p')

# # positivity pre 2020 is column "RSV Detections"/"RSV Tests"
# rsv_pre2020["Percent Positive"] = 100*rsv_pre2020["RSV Detections"]/rsv_pre2020["RSV Tests"]
# rsv_pre2020.fillna(0,inplace=True)
# # positivity post 2020 is column "pcr_percent_positive"
# rsv_post2020["Percent Positive"] = rsv_post2020["pcr_percent_positive"]

# # add "Region " string in formt of "HHS region " column in pre 2020 data
# rsv_pre2020["Region"] = "Region " + rsv_pre2020["HHS region "].astype(str)
# # drop "National" level from post 2020 data and rename
# rsv_post2020 = rsv_post2020[rsv_post2020["level"] != "National"]
# rsv_post2020.rename(columns={"level":"Region"},inplace=True)

# # concatenate time series
# rsv_pp = pd.concat([rsv_pre2020[["Date","Region","Percent Positive"]],rsv_post2020[["Date","Region","Percent Positive"]]])

# # average duplicates
# rsv_pp = rsv_pp.groupby(["Date","Region"]).mean().reset_index()

# # index and pivot
# rsv_pp_regional = rsv_pp.pivot(index="Date",columns="Region",values="Percent Positive")
# rsv_pp_regional = rsv_pp_regional[["Region " + str(i) for i in range(1,9)]+["Region 10"]+["Region 9"]]
# rsv_pp_regional.to_csv('Data/Processed/RSV_PercentPositive_Regions.csv')


# ### flu data
# flu_pre2015 = pd.read_csv('Data/Raw/FluViewPhase2Data_HHSRegions/WHO_NREVSS_Combined_prior_to_2015_16.csv')
# flu_clinical = pd.read_csv('Data/Raw/FluViewPhase2Data_HHSRegions/WHO_NREVSS_Clinical_Labs.csv')
# flu_ph = pd.read_csv('Data/Raw/FluViewPhase2Data_HHSRegions/WHO_NREVSS_Public_Health_Labs.csv')

# flu_pre2015["Date"] = pd.to_datetime(flu_pre2015["YEAR"].astype(int).astype(str) + '-01-01') + pd.to_timedelta(flu_pre2015["WEEK"]*7,unit='D')
# flu_clinical["Date"] = pd.to_datetime(flu_clinical["YEAR"].astype(int).astype(str) + '-01-01') + pd.to_timedelta(flu_clinical["WEEK"]*7,unit='D')

# # remove Pueto Rico, Virgin Islands, District of Columbia, New York City, and Rhode Island
# # flu_pre2015 = flu_pre2015[~flu_pre2015["REGION"].isin(["Puerto Rico","Virgin Islands","New York City"])]
# # flu_clinical = flu_clinical[~flu_clinical["REGION"].isin(["Puerto Rico","Virgin Islands","New York City"])]

# # replace "X" with NA
# flu_pre2015.replace("X",0.0,inplace=True)
# flu_clinical.replace("X",0.0,inplace=True)

# # flu A columns are A (2009 H1N1),A (H1),A (H3),A (Subtyping not Performed),A (Unable to Subtype),H3N2v,A (H5)
# flu_pre2015["A"] = flu_pre2015["A (2009 H1N1)"].astype(float) + flu_pre2015["A (H1)"].astype(float) + flu_pre2015["A (H3)"].astype(float) + flu_pre2015["A (Subtyping not Performed)"].astype(float) + flu_pre2015["A (Unable to Subtype)"].astype(float) + flu_pre2015["H3N2v"].astype(float) + flu_pre2015["A (H5)"].astype(float)
# flu_pre2015["PERCENT A"] = 100*flu_pre2015["A"]/flu_pre2015["TOTAL SPECIMENS"].astype(float)
# flu_pre2015["PERCENT B"] = 100*flu_pre2015["B"].astype(float)/flu_pre2015["TOTAL SPECIMENS"].astype(float)
# # NA to 0
# flu_pre2015.fillna(0,inplace=True)

# # get PERCENT POSITIVE for each week by concatenating time series from pre-2015 and post-2015 clinical data
# fluA_pp = pd.concat([flu_pre2015[["Date","REGION","PERCENT A"]],flu_clinical[["Date","REGION","PERCENT A"]]])
# fluA_pp.rename(columns={"PERCENT A":"PERCENT POSITIVE"},inplace=True)
# fluB_pp = pd.concat([flu_pre2015[["Date","REGION","PERCENT B"]],flu_clinical[["Date","REGION","PERCENT B"]]])
# fluB_pp.rename(columns={"PERCENT B":"PERCENT POSITIVE"},inplace=True)
# # index
# fluA_pp_regional = fluA_pp.pivot(index="Date",columns="REGION",values="PERCENT POSITIVE")
# # convert values type into float
# fluA_pp_regional = fluA_pp_regional.astype(float)
# fluA_pp_regional = fluA_pp_regional[["Region " + str(i) for i in range(1,9)]+["Region 10"]+["Region 9"]]
# fluB_pp_regional = fluB_pp.pivot(index="Date",columns="REGION",values="PERCENT POSITIVE")
# fluB_pp_regional = fluB_pp_regional.astype(float)
# fluB_pp_regional = fluB_pp_regional[["Region " + str(i) for i in range(1,9)]+["Region 10"]+["Region 9"]]
# fluA_pp_regional.to_csv('Data/Processed/FluView_PercentPositive_Regions_A.csv')
# fluB_pp_regional.to_csv('Data/Processed/FluView_PercentPositive_Regions_B.csv')


# # # plots
# # # colors = colormaps.get_cmap('Greys',9)(jnp.linspace(1,0.3,9)).tolist()
# # # colors.append('red')
# # fig, ax = plt.subplots(2,1,figsize=(13.3,7.5),sharey=True,sharex=True)
# # fluA_pp_regional.plot(ax=ax[0],legend=False,color='k',alpha=0.3)
# # fluB_pp_regional.plot(ax=ax[1],legend=False,color='k',alpha=0.3)
# # ax[0].set_ylabel("Percent positive for flu A")
# # ax[1].set_ylabel("Percent positive for flu B")
# # # legend label is State
# # # ax[1].legend(title="State")
# # fig.suptitle("FluView Percent Positive by State")
# # plt.tight_layout()
# # plt.savefig('Figures/FluView_PercentPositive_States_ppt.png',dpi=300)

# # plots
# # # plot the region between 2022-01-01 and 2022-06-01
# # # colors = ["#648FFF", "#DC267F", "#FFB000", "#785EF0", "#000000", "#648FFF", "#DC267F", "#FFB000", "#785EF0", "#000000"]
# # # linestyles = ['-']*5+['--']*5
# # linestyles = ['-']*10
# # fig, ax = plt.subplots(1,1,figsize=(13.3,7.5),sharey=True,sharex=True)
# # for i,region in enumerate(rsv_pp_regional.columns):
# #     rsv_pp_regional[region].plot(ax=ax, color=colors[i], linestyle=linestyles[i], label=region)
# # # fluA_pp_regional.loc['2022-01-01':'2022-06-01'].plot(ax=ax[0],color=colors,legend=False)
# # ax.legend(title="HHS region")
# # ax.set_ylabel("Percent positive for flu A")
# # plt.tight_layout()
# # plt.savefig('Figures/RSV_PercentPositive_Region_Highlight_ppt.png',dpi=300)

# ############## Plotting KPSC data ###############
# # Incidence line plots
# # fig, axes = plt.subplots(3,2,figsize=(13.3,7.5),sharex=True)
# # kpsc_positive_test_plot(axes[0,0],pathogen="Metapneumovirus", hospitalizations=True, incidence=True, legend=False,aggregation="Month",color='#648FFF')
# # kpsc_positive_test_plot(axes[1,0],pathogen="Adenovirus", hospitalizations=True, incidence=True, legend=False,aggregation="Month",color='#648FFF')
# # kpsc_positive_test_plot(axes[2,0],pathogen="Parainfluenza 3", hospitalizations=True, incidence=True, legend=False,aggregation="Month",color='#648FFF')
# # kpsc_positive_test_plot(axes[0,1],pathogen="Metapneumovirus",AGE_GROUPS=AGE_GROUPS,AGE_GROUP_NAMES=AGE_GROUP_NAMES, hospitalizations=True, incidence=True, legend=False,aggregation="Month")
# # kpsc_positive_test_plot(axes[1,1],pathogen="Adenovirus",AGE_GROUPS=AGE_GROUPS,AGE_GROUP_NAMES=AGE_GROUP_NAMES, hospitalizations=True, incidence=True, legend=False,aggregation="Month")
# # kpsc_positive_test_plot(axes[2,1],pathogen="Parainfluenza 3",AGE_GROUPS=AGE_GROUPS,AGE_GROUP_NAMES=AGE_GROUP_NAMES, hospitalizations=True, incidence=True, legend=True,aggregation="Month")
# # plt.tight_layout()
# # plt.savefig('Figures/KPSC_data_sort_of_interesting_slide.png',dpi=300)

# # Cumulative seasons plot
# plt.rcParams.update({'font.size':14})
# # text type is palatino
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Palatino']
# fig, axes = plt.subplots(2,6,sharex=True,figsize=(13.3,5))
# for row,pathogen in enumerate(["InfluenzaA","InfluenzaB","RSV","Metapneumovirus","Adenovirus","Parainfluenza3"]):
#     for col,typ in enumerate([[False,False],[False,True]]):
#         season_plot(axes[col,row],pathogen,typ[0],typ[1])
# # label every other x tick with season year
# axes[1,5].set_xticks(range(11),[f"20{year}/{year+1}" if year % 2 == 0 else "" for year in range(15,26)])
# # for i in range(6):
# #     for j in range(1,2):
# #         axes[j,i].set_yticks([],[])
# axes[0,0].set_ylabel("Cases")
# axes[1,0].set_ylabel("Age group share")
# axes[0,0].set_title("Influenza A")
# axes[0,1].set_title("Influenza B")
# axes[0,2].set_title("RSV")
# axes[0,3].set_title("Metapneumovirus")
# axes[0,4].set_title("Adenovirus")
# axes[0,5].set_title("Parainfluenza 3")
# # fig.suptitle("Cumulative cases by respiratory season")
# plt.tight_layout()
# plt.savefig("Figures/cumulative_seasons_extended.png",dpi=300)

# # ############## Processing KPSC demographic data ###############
# demo = pd.read_sas("Data/Raw/KPSC/demographics_20260128.sas7bdat", format='sas7bdat', encoding='utf-8')

# pop_by_age_group_month = demo.pivot_table(index="month_start", columns="age_category", values="n_count", aggfunc='sum').fillna(0)

# # # Create a mapping dictionary from original column names to AGE_GROUP_NAMES
# # column_mapping = dict(zip(['<3mo', '3-12mo', '1-4y', '5-7y', '8-39y', '40-64y', '>=65y'], AGE_GROUP_NAMES))
# # pop_by_age_group_month = pop_by_age_group_month.rename(columns=column_mapping)
# # pop_by_age_group_month = pop_by_age_group_month[AGE_GROUP_NAMES]

# # # save to csv
# # pop_by_age_group_month.to_csv('Data/Processed/KPSC_population_by_age_group_monthly.csv')

# # # ############### Processing KPSC data into time series of test-confirmed cases ###############



    # OLD CODE BELOW
    # # plot incidence per 10k for each pathogen over time in a six panel plot
    # fig, ax = plt.subplots(3,2,figsize=(13.3,7.5),sharex=True)
    # pop_by_age_group_month = pd.read_csv('Data/Processed/KPSC_population_by_age_group_monthly.csv', index_col=0, parse_dates=['month_start'])
    # pop_by_age_group_month = pop_by_age_group_month.reindex(columns=AGE_GROUP_NAMES)
    # pop_by_age_group_daily = pop_by_age_group_month.resample('D').ffill()
    
    # for pi,pathogen in enumerate(["InfluenzaA","InfluenzaB","RSV","Metapneumovirus","Adenovirus","Parainfluenza3"]):
    #     pathogen_data = pd.read_csv(f'Data/Processed/KPSC_unsalvage_panel_positive_{pathogen}_matched_noncovid_ARI_hospitalizations.csv', parse_dates=["Hospitalization date"])
    #     pathogen_data["Hospitalization date"] = pd.to_datetime(pathogen_data["Hospitalization date"])
    #     pathogen_data = pathogen_data.set_index("Hospitalization date")
        
    #     # Calculate daily incidence per 10k population
    #     daily_pop = pop_by_age_group_daily.reindex(pathogen_data.index).fillna(method='ffill')
    #     daily_incidence = (pathogen_data[AGE_GROUP_NAMES].sum(axis=1) / daily_pop[AGE_GROUP_NAMES].sum(axis=1)) * 10000
        
    #     # Resample to weekly and sum
    #     weekly_incidence = daily_incidence.resample("W").sum().reset_index()
    #     weekly_incidence.columns = ["Hospitalization date", "Incidence per 10k"]
        
    #     ax[pi//2, pi%2].plot(weekly_incidence["Hospitalization date"], weekly_incidence["Incidence per 10k"], color='k')
    #     ax[pi//2, pi%2].set_title(f"{pathogen}")
    #     ax[pi//2, 0].set_ylabel('Incidence per 10k')
    #     ax[1, pi%2].set_xticks(weekly_incidence["Hospitalization date"][::52], [str(year) for year in weekly_incidence["Hospitalization date"].dt.year[::52]], rotation=45)
    # plt.tight_layout()
    # plt.savefig("Figures/KPSC_unsalvage_panel_positive_matched_noncovid_ARI_hospitalizations_incidence_weekly_test.png",dpi=300)


# #### Plot number of RSV tests and proportion of respiratory clinical cases with RSV tests over time
# #age filter
# tests_data = test_data[test_data["age_in_mo"] < 12*18]
# clinical_data = clinical_data[clinical_data["age_in_mo"] < 12*18]
# tests_data = test_data[test_data["age_in_mo"] >= 12*5]
# clinical_data = clinical_data[clinical_data["age_in_mo"] >= 12*5]
# # other filters
# tests_RSV = test_data[test_data["pathogen"].str.contains("RESPIRATORY SYNCYTIAL VIRUS",na=False)]
# tests_RSV["Date"] = pd.to_datetime(tests_RSV["YEAR"].astype(int).astype(str) + '-10-01') + pd.to_timedelta(tests_RSV["lab_days"],unit='D')
# # plot number of RSV tests over time
# tests_RSV_datesums = pd.pivot_table(tests_RSV, index='Date', columns='result_val', values='StudyID', aggfunc='count').fillna(0)
# fig, ax = plt.subplots(2,1,figsize=(6.5,6.5), sharex=True)
# tests_RSV_datesums.sum(axis=1).plot(ax=ax[0], color='black',label='Total')
# # also plot positive tests
# tests_RSV_datesums['Positive'].plot(ax=ax[0], color='red',label='Positive')
# ax[0].set_title("Number of RSV tests over time")
# ax[0].legend()
# ax[0].set_ylabel("Number of tests")
# ax[0].set_xlabel("Date")

# respiratory_codes = pd.read_csv('Data/Processed/respiratory_codes.csv',dtype=str)
# respiratory_clinical_data = clinical_data[clinical_data["CODE"].isin(respiratory_codes)]
# respiratory_clinical_data["Date"] = pd.to_datetime(respiratory_clinical_data["YEAR"].astype(int).astype(str) + '-10-01') + pd.to_timedelta(respiratory_clinical_data["dx_days"],unit='D')
# respiratory_datesums = pd.pivot_table(respiratory_clinical_data, index="Date", values="StudyID", aggfunc='count').fillna(0)
# proportion_rsv_tests = pd.merge(respiratory_datesums, tests_RSV_datesums, left_index=True, right_index=True, how='left').fillna(0)
# proportion_rsv_tests["Proportion RSV tests"] = proportion_rsv_tests[["Invalid","Negative","Positive"]].sum(axis=1)/proportion_rsv_tests["StudyID"]
# proportion_rsv_tests["Proportion RSV tests"].plot(ax=ax[1], color='blue')
# ax[1].set_title("Proportion of respiratory clinical cases with RSV tests")
# ax[1].set_ylabel("Proportion of clinical cases with RSV tests")
# ax[1].set_xlabel("Date")
# plt.savefig('Figures/KPSC_RSV_tests_over_time.png',dpi=300)



# # # save random sample of clinical data
# # # clinical_data.sample(10000).to_csv('Data/Processed/KPSC_clinical_sample.csv',index=False)
# # # # load
# # # clinical_data = pd.read_csv('Data/Processed/KPSC_clinical_sample.csv')
# # # # date is 1st of October of each year (in YEAR column), plus dx_days

# non_ari = clinical_data[clinical_data["dxgroup"] != "ARI"]

# # # get proportion of clinical cases with flu_vac == 1 in each month, for each age group.
# vaccination_proportion = clinical_data.groupby(["Month","AGE_GROUP"], observed=True)["flu_vac"].mean().unstack()
# # vaccination_proportion = vaccination_proportion.reindex(pd.period_range(start=vaccination_proportion.index.min(),end=vaccination_proportion.index.max(),freq='M'))
# # print(vaccination_proportion)
# vaccination_proportion = vaccination_proportion.fillna(0)
# #reorder columns to match order in AGE_GROUP_NAMES
# vaccination_proportion = vaccination_proportion[AGE_GROUP_NAMES]

# # # # save to csv
# vaccination_proportion.to_csv('Data/Processed/KPSC_vaccinated_proportion_ages_monthly.csv')

# # #load
# # vaccination_proportion = pd.read_csv('Data/Processed/KPSC_vaccinated_proportion_ages_by_season.csv',index_col=0)
# vax2022 = vaccination_proportion.loc['2022-10-01']
# age_pops = pd.read_csv('Data/Raw/US_Census_population_by_age.csv',dtype=int)
# age_pop = age_pops.groupby('AGE')['POPESTIMATE2022'].sum()
# # get rid of age over 90
# age_pop = age_pop[age_pop.index < 90]
# # match vaccination proportion to age population by age
# vax2022 = vax2022.reindex(age_pop.index)
# # population weighted average of vaccination proportion
# vax2022 = (vax2022*age_pop).sum()/age_pop.sum()
# print(vax2022)

# print(vaccination_proportion[['<3m','3-11m']])
# # plot
# fig, ax = plt.subplots(figsize=(6.5,6.5))
# hsv_colors = colormaps.hsv(-0.02+jnp.arange(7)/7)
# hsv_colors[3] = colormaps.hsv((3/7)+0.04)
# vaccination_proportion.plot(ax=ax,color=hsv_colors)
# # # plot dashed vertical lines at october each year
# # for year in range(9):
# #     ax.axvline(12*year,color='black',alpha=0.3)
# ax.set_ylim(0,1)
# # # x labels based on years - first index is october 2015
# # ax.set_xticks(range(3,len(vaccination_proportion),12),[year for year in range(2016,2024)])
# # label seasons, e.g. 2015/16, 2016/17, etc.
# ax.set_xticks(range(8),[f"20{year}/{year+1}" for year in range(15,23)])
# ax.set_xlim(0,7)
# ax.set_title("Proportion of clinical cases with recent (<1y) flu vaccine")
# ax.set_ylabel("Proportion")
# ax.set_xlabel("Season")
# plt.savefig('Figures/KPSC_vaccinated_proportion_age_by_season.png',dpi=300)

# test_data = test_data[test_data["StudyID"].isin(clinical_data["StudyID"])].copy()
# positive_tests = test_data[test_data["result_val"] == 'Positive'].copy()
# positive_tests.loc[:,"Date"] = pd.to_datetime(positive_tests["YEAR"].astype(int).astype(str) + '-10-01') + pd.to_timedelta(positive_tests["lab_days"].astype(int),unit='D')
# positive_tests = clinical_data.merge(positive_tests[["StudyID","Date","pathogen","lab_type","lab_days"]],on="StudyID",how="left")
# positive_tests = positive_tests.rename(columns={"Date_x":"Clinical date","Date_y":"Test date"})
# positive_tests = positive_tests[np.abs((pd.to_datetime(positive_tests["Clinical date"]) - pd.to_datetime(positive_tests["Test date"])).dt.days) <= 14]

# # # matching_tests = test_data[test_data["StudyID"].isin(clinical_data["StudyID"])]
# # # matching_tests["Date"] = pd.to_datetime(matching_tests["YEAR"].astype(int).astype(str) + '-10-01') + pd.to_timedelta(matching_tests["lab_days"],unit='D')
# # # matching_tests = clinical_data.merge(matching_tests[["StudyID","Date","pathogen","lab_type","lab_days","result_val"]],on="StudyID",how="left")
# # # matching_tests = matching_tests.rename(columns={"Date_x":"Clinical date","Date_y":"Test date"})
# # # matching_tests = matching_tests[jnp.abs((pd.to_datetime(matching_tests["Clinical date"]) - pd.to_datetime(matching_tests["Test date"])).dt.days) <= 14]

# # # clear unused test data from memory
# # del test_data

# # # save to csv
# positive_tests.to_csv('Data/Processed/KPSC_unsalvage_noncovid_ARI_positive_matched_all_clinical.csv',index=False)
# # load
# # positive_tests = pd.read_csv('Data/Processed/KPSC_positive_matched_all_clinical.csv')
# hospitalizations = clinical_data[(clinical_data["setting"] == 'Hospital admission')]
# # # # clear unused clinical data from memory
# # del clinical_data

# # # # save hostpitalizations to csv
# hospitalizations.to_csv('Data/Processed/KPSC_unsalvage_noncovid_ARI_clinical_hospitalizations.csv',index=False)
# # # # load
# # hospitalizations = pd.read_csv('Data/Processed/KPSC_clinical_hospitalizations.csv')

# # respiratory_codes = pd.read_csv('Data/Processed/respiratory_codes.csv',dtype=str)
# # gastroenteritis_codes = pd.read_csv('Data/Processed/gastroenteritis_codes.csv',dtype=str)
# # # # get only hospitalizations with respiratory or gastroenteritis codes
# respiratory_hospitalizations = hospitalizations[hospitalizations["dxgroup"] == "ARI"]
# gastroenteritis_hospitalizations = hospitalizations[hospitalizations["dxgroup"] != "ARI"]
# # # # # save
# respiratory_hospitalizations.to_csv('Data/Processed/KPSC_clinical_ARI_hospitalizations.csv',index=False)
# gastroenteritis_hospitalizations.to_csv('Data/Processed/KPSC_clinical_gastroenteritis_hospitalizations.csv',index=False)
# # # respiratory_hospitalizations = pd.read_csv('Data/Processed/KPSC_clinical_respiratory_hospitalizations.csv')
# # # gastroenteritis_hospitalizations = pd.read_csv('Data/Processed/KPSC_clinical_gastroenteritis_hospitalizations.csv')

# # filter test data to only include tests with StudyID matching a value in hospitalizations
# test_data = test_data[test_data["StudyID"].isin(hospitalizations["StudyID"])]
# test_data["Date"] = pd.to_datetime(test_data["YEAR"].astype(int).astype(str) + '-10-01') + pd.to_timedelta(test_data["lab_days"],unit='D')
# # # # save test data
# test_data.to_csv('Data/Processed/KPSC_hospitalized_tests.csv',index=False)
# # # # load test data
# # test_data = pd.read_csv('Data/Processed/KPSC_hospitalized_tests.csv')
# # positive_tests = test_data[test_data["result_val"] == 'Positive']


###### find hospitalizations with matching positive tests ########
# test_data = test_data[test_data["StudyID"].isin(hospitalizations["StudyID"])].copy()
# positive_tests = test_data[test_data["result_val"] == 'Positive'].copy()
# positive_tests.loc[:,"Date"] = pd.to_datetime(positive_tests["YEAR"].astype(int).astype(str) + '-10-01') + pd.to_timedelta(positive_tests["lab_days"],unit='D')
# # # find date of from hospitalizations for each study ID and match to positive tests
# positive_tests = hospitalizations.merge(positive_tests[["StudyID","Date","pathogen","lab_type","lab_days"]],on="StudyID",how="left")
# # # # rename columns
# positive_tests = positive_tests.rename(columns={"Date_x":"Hospitalization date","Date_y":"Test date"})
# # # # keep only rows where Hospitalization date is within 14 days of Test date
# positive_tests = positive_tests[np.abs((pd.to_datetime(positive_tests["Hospitalization date"]) - pd.to_datetime(positive_tests["Test date"])).dt.days) <= 14]
# # keep only one test per pathogen and hospitalization
# positive_tests = positive_tests.sort_values(by=["StudyID","pathogen","Test date"], ascending=[True,True,True])
# # find groups of tests within 14 days of each other with the same StudyID and pathogen
# positive_tests.loc[:,"diff"] = positive_tests.groupby(["StudyID","pathogen"])["Test date"].diff().dt.days
# # for groups of tests where diff is less than 14 days, keep only the first test
# positive_tests = positive_tests[(positive_tests["diff"].isna()) | (positive_tests["diff"] > 14)]

# # save to csv
# positive_tests.to_csv('Data/Processed/KPSC_unsalvage_noncovid_ARI_positive_matched_hospitalizations.csv',index=False)


# #### Create daily counts of hospitalized test results ########
# test_data = load_and_filter_test_data()

# hospitalizations = pd.read_csv('Data/Processed/KPSC_clinical_hospitalizations.csv')
# resp_hospitalizations = hospitalizations[hospitalizations["dxgroup"] == "ARI"]

# # Create daily counts keeping only one test per pathogen and hospitalization
# all_tests = test_data[test_data["StudyID"].isin(resp_hospitalizations["StudyID"])].copy()
# all_tests.loc[:,"Date"] = (pd.to_datetime(all_tests["YEAR"].astype(int).astype(str) + '-10-01') + pd.to_timedelta(all_tests["lab_days"].astype(int),unit='D')).dt.normalize()
# resp_hospitalizations.loc[:,"Date"] = (pd.to_datetime(resp_hospitalizations["YEAR"].astype(int).astype(str) + '-10-01') + pd.to_timedelta(resp_hospitalizations["dx_days"].astype(int),unit='D')).dt.normalize()

# # Merge with hospitalization data to get hospitalization dates
# all_tests = resp_hospitalizations.merge(all_tests[["StudyID","Date","pathogen","lab_type","lab_days","result_val"]],on="StudyID",how="left")
# all_tests = all_tests.rename(columns={"Date_x":"Hospitalization date","Date_y":"Test date"})

# # Classify age groups
# from Parameters.census_population import AGE_GROUPS, AGE_GROUP_NAMES
# bins = [group[0] for group in AGE_GROUPS] + [AGE_GROUPS[-1][-1] + 1]
# all_tests.loc[:,"age_group"] = pd.cut(all_tests["age_in_mo"], bins=bins, labels=AGE_GROUP_NAMES, right=False)

# # Keep only tests within 14 days of hospitalization
# all_tests = all_tests[np.abs((pd.to_datetime(all_tests["Hospitalization date"]) - pd.to_datetime(all_tests["Test date"])).dt.days) <= 14]

# pathogen_names = {
# "RSV": ["RESPIRATORY SYNCYTIAL VIRUS","RESPIRATORY SYNCYTIAL VIRUS SUBTYPE A","RESPIRATORY SYNCYTIAL VIRUS SUBTYPE B",],
# "InfluenzaA": ["INFLUENZA A","INFLUENZA A H1N1 2009","INFLUENZA A VIRUS","INFLUENZA A VIRUS SUBTYPE H1","INFLUENZA A VIRUS SUBTYPE/HEMAGGLUTININ H3","INFLUENZA VIRUS A",],
# "InfluenzaB": ["INFLUENZA B","INFLUENZA VIRUS B",],
# "Metapneumovirus": ["HUMAN METAPNEUMOVIRUS VIRUS",],
# "Adenovirus": ["ADENOVIRUS",],
# "Parainfluenza1": ["PARAINFLUENZA VIRUS 1"],
# "Parainfluenza2": ["PARAINFLUENZA VIRUS 2"],
# "Parainfluenza3": ["PARAINFLUENZA VIRUS 3"],
# "Parainfluenza4": ["PARAINFLUENZA VIRUS 4"],
# "Rhinovirus": ["ENTEROVIRUS/RHINOVIRUS"],
# "Pertussis": ["BORDETELLA PERTUSSIS"],
# "M.pneumoniae": ["MYCOPLASMA PNEUMONIAE"],
# "C.pneumoniae": ["CHLAMYDOPHILA PNEUMONIAE"],
# "SARS-CoV-2": ["SARS-COV-2 (COVID-19)"],
# "Enterovirus": ["ENTEROVIRUS/RHINOVIRUS"],
# }
# # replace pathogen names with group names
# # Create a single mapping dictionary from all pathogen names to group names
# reverse_names = [{v:k for v in values} for k,values in pathogen_names.items()]
# reverse_names =  {k:v for d in reverse_names for k,v in d.items()}
# all_tests.loc[:,"pathogen"] = all_tests["pathogen"].map(reverse_names)
# all_tests = all_tests[all_tests["pathogen"].notna()]

# pathogen_list = ["InfluenzaA","InfluenzaB","RSV","Metapneumovirus","Adenovirus","Parainfluenza3"]
# # Keep only panel tests, i.e. there is a test for each pathogen in the list (that is, containing the string for each pathogen in the list in the pathogen column) for a given StudyID and test date
# panel_test_groups = all_tests.groupby(["StudyID","Test date"])["pathogen"].apply(lambda x: all(any(p == pathogen for pathogen in x) for p in pathogen_list))
# all_tests = all_tests.set_index(["StudyID","Test date"]).loc[panel_test_groups[panel_test_groups].index].reset_index()

# # Keep only one test per pathogen and hospitalization - prioritize positive tests
# all_tests = all_tests.sort_values(by=["StudyID","pathogen","result_val","Test date"], ascending=[True,True,False,True])
# all_tests.loc[:,"diff"] = all_tests.groupby(["StudyID","pathogen"])["Test date"].diff().dt.days
# all_tests = all_tests[(all_tests["diff"].isna()) | (all_tests["diff"] > 14)]

# # Create daily aggregations for total tests and positive tests
# daily_total_counts = all_tests.groupby(['Hospitalization date', 'pathogen', 'age_group']).size().reset_index(name='count')
# daily_total_counts.loc[:,'result_type'] = 'Total'

# daily_positive_counts = all_tests[all_tests['result_val'] == 'Positive'].groupby(['Hospitalization date', 'pathogen', 'age_group']).size().reset_index(name='count')
# daily_positive_counts.loc[:,'result_type'] = 'Positive'

# # Combine total and positive counts
# daily_test_counts = pd.concat([daily_total_counts, daily_positive_counts], ignore_index=True)

# # Create complete date range and pathogen/result combinations
# date_range = pd.date_range(start=daily_test_counts['Hospitalization date'].min(), 
#                           end=daily_test_counts['Hospitalization date'].max(), 
#                           freq='D')
# all_pathogens = daily_test_counts['pathogen'].unique()
# all_age_groups = daily_test_counts['age_group'].unique()
# result_types = ['Total', 'Positive']

# # Create complete combinations
# complete_combinations = pd.MultiIndex.from_product([date_range, all_pathogens, all_age_groups, result_types], 
#                                                   names=['Hospitalization date', 'pathogen', 'age_group', 'result_type']).to_frame(index=False)

# # Merge with actual counts and fill zeros
# daily_test_counts_complete = complete_combinations.merge(daily_test_counts, 
#                                                         on=['Hospitalization date', 'pathogen', 'age_group', 'result_type'], 
#                                                         how='left')
# daily_test_counts_complete.loc[:,'count'] = daily_test_counts_complete['count'].fillna(0).astype(int)

# daily_test_counts_complete.to_csv('Data/Processed/KPSC_ARI_hospitalized_pathogen_unsalvage_panel_test_counts_by_hosp_day_pathogen_age_group.csv', index=False)

# #load test counts
# daily_test_counts_complete = pd.read_csv('Data/Processed/KPSC_ARI_hospitalized_pathogen_unsalvage_panel_test_counts_by_hosp_day_pathogen_age_group.csv')

# # Filter to only include pathogens that are in our mapping and sum by group
# grouped_data = daily_test_counts_complete.groupby(['Hospitalization date', 'pathogen', 'age_group', 'result_type'])['count'].sum().reset_index()

# # Create six panel plot separating test data for each age group
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(4, 2, figsize=(13.3, 7.5), sharex=True)
# ax = ax.flatten()  # Flatten for easier indexing

# select_age_groups = AGE_GROUP_NAMES

# for i, age_group in enumerate(select_age_groups):
#     age_data = grouped_data[(grouped_data['age_group'] == age_group) & (grouped_data['result_type'] == 'Total')]
#     age_daily_counts = age_data.groupby('Hospitalization date')['count'].sum().reset_index()
#     age_daily_counts['Hospitalization date'] = pd.to_datetime(age_daily_counts['Hospitalization date'])
#     age_weekly_counts = age_daily_counts.set_index('Hospitalization date').resample('W')['count'].sum().reset_index()
    
#     ax[i].plot(age_weekly_counts['Hospitalization date'], age_weekly_counts['count'], 
#                color='k', linewidth=1)
    
#     ax[i].set_title(f"{age_group}")
#     if i >= 4:  # Bottom row
#         ax[i].set_xlabel('Date')
#     if i % 2 == 0:  # Left column
#         ax[i].set_ylabel('Number of tests')

# plt.suptitle("Weekly total hospitalized panel test counts for ARI by age group")
# plt.tight_layout()
# plt.savefig('Figures/KPSC_ARI_hospitalized_pathogen_unsalvage_panel_test_weekly_counts_by_age_group_panels.png', dpi=300)
# plt.close()

# ###### Total number of ARI hospitalizations each day ########
# hospitalizations = pd.read_csv('Data/Processed/KPSC_clinical_hospitalizations.csv')
# resp_hospitalizations = hospitalizations[hospitalizations["dxgroup"] == "ARI"]
# resp_hospitalizations.loc[:,"Hospitalization date"] = pd.to_datetime(resp_hospitalizations["YEAR"].astype(int).astype(str) + '-10-01') + pd.to_timedelta(resp_hospitalizations["dx_days"],unit='D')

# # keep only one record for each hospitalization per StudyID within 14 days
# resp_hospitalizations = resp_hospitalizations.sort_values(by=["StudyID","Hospitalization date"], ascending=[True,True])
# resp_hospitalizations.loc[:,"diff"] = resp_hospitalizations.groupby(["StudyID"])["Hospitalization date"].diff().dt.days
# resp_hospitalizations = resp_hospitalizations[(resp_hospitalizations["diff"].isna()) | (resp_hospitalizations["diff"] > 14)]

# # Exclude records where the same StudyID has a COVID-19 diagnosis (U07.1) within a 14-day window
# covid_records = resp_hospitalizations[resp_hospitalizations["CODE"] == "U07.1"].copy()
# # print(covid_records.head())
# # what is the earliest date in covid_records
# print("Earliest COVID record date: ", covid_records["Hospitalization date"].min())

# if not covid_records.empty:
#     # For each record, check if there's a COVID record for the same StudyID within 14 days
#     exclude_indices = []
#     total_records = len(resp_hospitalizations)
    
#     for i, (idx, row) in enumerate(resp_hospitalizations.iterrows()):
#         if i % 10000 == 0:  # Print progress every 10,000 records
#             print(f"Processing record {i+1}/{total_records} ({(i+1)/total_records*100:.1f}%)")
        
#         study_id = row["StudyID"]
#         hosp_date = row["Hospitalization date"]
        
#         # Get all COVID dates for this StudyID
#         covid_dates = covid_records[covid_records["StudyID"] == study_id]["Hospitalization date"]
        
#         if not covid_dates.empty:
#             # Calculate days difference with all COVID dates for this StudyID
#             days_diffs = np.abs((pd.to_datetime(hosp_date) - pd.to_datetime(covid_dates)).dt.days)
            
#             # If any COVID date is within 14 days, exclude this record
#             if (days_diffs <= 14).any():
#                 exclude_indices.append(idx)
    
#     print(f"Excluding {len(exclude_indices)} records with COVID diagnoses within 14 days")
#     # Remove excluded records
#     resp_hospitalizations = resp_hospitalizations.drop(exclude_indices)

# # Classify age groups
# bins = [group[0] for group in AGE_GROUPS] + [AGE_GROUPS[-1][-1] + 1]
# resp_hospitalizations.loc[:,"age_group"] = pd.cut(resp_hospitalizations["age_in_mo"], bins=bins, labels=AGE_GROUP_NAMES, right=False)

# # ### no age group version
# # daily_hospitalization_counts = resp_hospitalizations.groupby('Hospitalization date').size().reset_index(name='count')

# ## age group version
# daily_hospitalization_counts = resp_hospitalizations.pivot_table(index='Hospitalization date', columns='age_group', values='StudyID', aggfunc='count').fillna(0).reset_index()
# # reorder columns
# daily_hospitalization_counts = daily_hospitalization_counts[['Hospitalization date'] + AGE_GROUP_NAMES]
# daily_hospitalization_counts.to_csv('Data/Processed/KPSC_ARI_nonCOVID_hospitalizations_by_day_age_group.csv', index=False)

# # print most common day
# print(daily_hospitalization_counts.sort_values(by='count', ascending=False).head())
# # get second highest day
# second_max = daily_hospitalization_counts.sort_values(by='count', ascending=False).iloc[1]

# # plot daily hospitalization counts on single axis
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(figsize=(6.5, 4))

# # plot total daily hospitalizations
# daily_hospitalization_counts.set_index('Hospitalization date')['count'].plot(ax=ax, color='black', label='Total')

# ax.set_title("Daily ARI hospitalizations KPSC, without repeat records within 14 days")
# ax.set_ylabel("Number of hospitalizations")
# ax.set_xlabel("Date")

# plt.tight_layout()
# plt.savefig('Figures/KPSC_ARI_hospitalizations_nonduplicate_by_day.png', dpi=300)


################ Does testing behaviour change over time? ################

# matching_tests = respiratory_hospitalizations.merge(matching_tests[["StudyID","Date","pathogen","lab_type","lab_days","result_val"]],on="StudyID",how="left")
# # # # rename columns
# matching_tests = matching_tests.rename(columns={"Date_x":"Hospitalization date","Date_y":"Test date"})
# from plotting import reverse_names as pathogen_names
# # replace subtyped or detailed pathogen names with main pathogen name
# matching_tests["pathogen"] = matching_tests["pathogen"].replace(pathogen_names)
# print(matching_tests["pathogen"].value_counts())

# # respiratory failure as a proxy for most severe cases
# severe_tests = matching_tests[matching_tests["CODE"].str.contains("J96",na=False)]
# print(severe_tests.head())
# severe_tests = severe_tests[np.abs((pd.to_datetime(severe_tests["Hospitalization date"]) - pd.to_datetime(severe_tests["Test date"])).dt.days) <= 14]
# severe_tests = severe_tests.sort_values(by=["StudyID","pathogen","result_val","Test date"], ascending=[True,True,False,True])
# severe_tests["diff"] = severe_tests.groupby(["StudyID","pathogen"])["Test date"].diff().dt.days
# severe_tests = severe_tests[(severe_tests["diff"].isna()) | (severe_tests["diff"] > 14)]

# # # # keep only rows where Hospitalization date is within 14 days of Test date
# matching_tests = matching_tests[np.abs((pd.to_datetime(matching_tests["Hospitalization date"]) - pd.to_datetime(matching_tests["Test date"])).dt.days) <= 14]
# # sorting by StudyID, pathogen, result_val (with Positive first), and Test date. Sorting result_val with Positive first means we keep the Positive test if there is one
# matching_tests = matching_tests.sort_values(by=["StudyID","pathogen","result_val","Test date"], ascending=[True,True,False,True])
# # find groups of tests within 14 days of each other with the same StudyID and pathogen
# matching_tests["diff"] = matching_tests.groupby(["StudyID","pathogen"])["Test date"].diff().dt.days
# # for groups of tests where diff is less than 14 days, keep only the first test
# matching_tests = matching_tests[(matching_tests["diff"].isna()) | (matching_tests["diff"] > 14)]

# # plot total number of matching tests over time
# matching_tests_RSV = matching_tests[matching_tests["pathogen"] == "RSV"]
# severe_tests_RSV = severe_tests[severe_tests["pathogen"] == "RSV"]
# print(severe_tests_RSV["CODE"].value_counts())

# matching_RSV_datesums = pd.pivot_table(matching_tests_RSV, index='Hospitalization date', columns='result_val', values='StudyID', aggfunc='count').fillna(0)
# severe_RSV_datesums = pd.pivot_table(severe_tests_RSV, index='Hospitalization date', columns='result_val', values='StudyID', aggfunc='count').fillna(0)
# print(severe_RSV_datesums.head())
# #aggregate by month
# matching_RSV_datesums.index = pd.to_datetime(matching_RSV_datesums.index)
# matching_RSV_datesums = matching_RSV_datesums.resample('M').sum()
# severe_RSV_datesums.index = pd.to_datetime(severe_RSV_datesums.index)
# severe_RSV_datesums = severe_RSV_datesums.resample('M').sum()

# fig, ax = plt.subplots(4,1,figsize=(6.5,8.5), sharex=True)
# matching_RSV_datesums.sum(axis=1).plot(ax=ax[0], color='black',label='Total')
# # also plot positive tests
# matching_RSV_datesums['Positive'].plot(ax=ax[0], color='red',label='Positive')
# ax[0].set_title("Number of tests among hospitalized cases over time")
# ax[0].legend()
# ax[0].set_ylabel("Number of tests")
# ax[0].set_xlabel("Date")

# # plot number of matching tests per hospitalization over time
# total_hospitalizations = respiratory_hospitalizations.groupby('Date').size()
# total_hospitalizations.index = pd.to_datetime(total_hospitalizations.index)
# total_hospitalizations = total_hospitalizations.resample('M').sum()
# proportion_matching_tests = matching_RSV_datesums.sum(axis=1)/total_hospitalizations
# proportion_matching_tests.plot(ax=ax[1], color='blue')
# ax[1].set_title("Proportion of respiratory hospitalized cases tested for RSV")
# ax[1].set_ylabel("Proportion tested")
# ax[1].set_xlabel("Date")

# # proportion of RSV-tested hospitalizations with respiratory failure
# proportion_severe_RSV = severe_RSV_datesums.sum(axis=1)/matching_RSV_datesums.sum(axis=1)
# proportion_severe_RSV.plot(ax=ax[2], color='blue')
# ax[2].set_title("Proportion of RSV-tested hospitalized cases with respiratory failure")
# ax[2].set_ylabel("Proportion with respiratory failure")
# ax[2].set_xlabel("Date")

# # proportion of RSV-positive hospitalizations with respiratory failure
# proportion_severe_RSV = severe_RSV_datesums["Positive"]/matching_RSV_datesums["Positive"]
# proportion_severe_RSV.plot(ax=ax[3], color='blue')
# ax[3].set_title("Proportion of RSV-positive hospitalized cases with respiratory failure")
# ax[3].set_ylabel("Proportion with respiratory failure")
# ax[3].set_xlabel("Date")

# fig.suptitle("School-aged children (5–17y))")

# plt.savefig('Figures/KPSC_hospitalized_SAC_RSV_tests_multi.png',dpi=300)


# positive_tests = pd.read_csv('Data/Processed/KPSC_positive_matched_hospitalizations.csv')
# # remove positive tests within 14 days of each other with the same StudyID and pathogen
# positive_tests = positive_tests.sort_values(by=["StudyID","pathogen","Test date"])
# positive_tests["Test date"] = pd.to_datetime(positive_tests["Test date"])
# positive_tests["Hospitalization date"] = pd.to_datetime(positive_tests["Hospitalization date"])
# # find groups of tests within 14 days of each other with the same StudyID and pathogen
# positive_tests["diff"] = positive_tests.groupby(["StudyID","pathogen"])["Test date"].diff().dt.days
# # change printing options to show extra rows
# pd.set_option('display.max_rows', 500)
# print(positive_tests.loc[positive_tests["StudyID"]==44833,["StudyID","pathogen","Test date","diff"]].head(100))
# # for groups of tests where diff is less than 14 days, keep only the first test
# positive_tests = positive_tests[(positive_tests["diff"].isna()) | (positive_tests["diff"] > 14)]

# positive_tests_not_cleaned = pd.read_csv('Data/Processed/KPSC_positive_matched_hospitalizations.csv')
# positive_tests_cleaned = pd.read_csv('Data/Processed/KPSC_positive_matched_hospitalizations_cleaned.csv')
# print(positive_tests.shape, positive_tests_not_cleaned.shape, positive_tests_cleaned.shape)
# print(positive_tests["StudyID"].nunique(), positive_tests_not_cleaned["StudyID"].nunique(), positive_tests_cleaned["StudyID"].nunique())
# print(positive_tests["pathogen"].value_counts())
# print(positive_tests_not_cleaned["pathogen"].value_counts())
# print(positive_tests_cleaned["pathogen"].value_counts())
# # shows that KPSC_positive_matched_hospitalizations_cleaned.csv has already been cleaned

## Separating out individual pathogen data from positive matched hospitalizations
# AGE_GROUP_NAMES = ['<3m','3-11m','1-4y','5-7y','8-39y','40-64y','>=65y']
# aggregation = None
# fig, ax = plt.subplots()
# for incidence in [False, True]:
#     for AGE_GROUPS in [None,[range(0,3), range(3,12),range(12,5*12),range(5*12,18*12),range(18*12,40*12),range(40*12,65*12),range(65*12,90*12)]]:
#         for pathogen in ["RSV","Influenza_A","Influenza_B","Metapneumovirus","Adenovirus","Parainfluenza 3"]:
#             kpsc_positive_test_plot(ax,pathogen=pathogen,AGE_GROUPS=AGE_GROUPS,AGE_GROUP_NAMES=AGE_GROUP_NAMES, hospitalizations=True, incidence=incidence, legend=True,aggregation=aggregation, save_data=True)
