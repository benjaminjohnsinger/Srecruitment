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

################ Data processing functions ################

##### function to calculate proportion positive tests for a given pathogen in a moving window, and multiply by population-proportional incidence of ARI hospitalizations ######
# daily_hospitalization_rates = pd.read_csv('Data/Processed/KPSC_ARI_hospitalization_rates_by_day_age_group.csv',index_col=0,parse_dates=True)

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

def calculate_proportion_positive_incidence(pathogen, window_size=28, weighting_factor=0.1, aggregation='D', sum_age_groups=False, cum_sum=False, save_counts=False, pp_only=False, hosp=False, salvage=True, pop_by_age_group_month=None, daily_hospitalization_counts=None, daily_test_counts_complete=None, return_counts=False, detrend=False, dedup=True, sac=True, NAG=7, censor=False, return_ci=False):
    if cum_sum and sum_age_groups:
        print("Warning: sum_age_groups ignored because cum_sum is True (the final column will represent the total sum).")
        sum_age_groups = False

    if sac and NAG>7:
        print("Warning: SAC age groups only defined for NAG=7. Setting sac=False.")
        sac = False
    
    if sac:
        from Parameters.census_population import AGE_GROUP_NAMES_sac as AGE_GROUP_NAMES
    elif NAG > 7:
        from Parameters.census_population import AGE_GROUP_NAMES_split as AGE_GROUP_NAMES
    else:
        from Parameters.census_population import AGE_GROUP_NAMES
    
    if pop_by_age_group_month is None:
        if NAG == 7:
            pop_by_age_group_month = pd.read_csv('Data/Processed/KPSC_population_by_age_group_monthly'+["","_sac"][sac]+'.csv', index_col=0, parse_dates=['month_start'])
        elif NAG > 7:
            pop_by_age_group_month = pd.read_csv('Data/Processed/KPSC_population_by_age_group_monthly_split.csv', index_col=0, parse_dates=['month_start'])
    if daily_hospitalization_counts is None:
        if NAG == 7:
            daily_hospitalization_counts = pd.read_csv('Data/Processed/KPSC_ARI_nonCOVID_hospitalizations_by_day_age_group'+['','_sac'][sac]+['','_dedup'][dedup]+'.csv', index_col=0, parse_dates=True)
        elif NAG > 7:
            daily_hospitalization_counts = pd.read_csv('Data/Processed/KPSC_ARI_nonCOVID_hospitalizations_by_day_age_group_split'+['','_dedup'][dedup]+['','_detrended'][detrend]+'.csv', index_col=0, parse_dates=True)
    if daily_test_counts_complete is None:
        if NAG == 7:
            daily_test_counts_complete = pd.read_csv('Data/Processed/KPSC_ARI_hospitalized_pathogen'+['','_unsalvage'][not salvage]+'_panel_test_counts_by'+['','_hosp'][hosp]+'_day_pathogen_age_group'+['','_sac'][sac]+['','_dedup'][dedup]+['','_EV'][pathogen=="Enterovirus"]+'.csv',index_col=0,parse_dates=True)
        elif NAG > 7:
            daily_test_counts_complete = pd.read_csv('Data/Processed/KPSC_ARI_hospitalized_pathogen'+['','_unsalvage'][not salvage]+'_panel_test_counts_by'+['','_hosp'][hosp]+'_day_pathogen_age_group_split'+['','_dedup'][dedup]+['','_EV'][pathogen=="Enterovirus"]+'.csv',index_col=0,parse_dates=True)

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

    # Calculate cumulative sums across age groups if requested
    if cum_sum:
        positive_counts = positive_counts.cumsum(axis=1)
        total_counts = total_counts.cumsum(axis=1)
        # Ensure base dataframes are properly aligned to age groups before cumulatively summing
        pop_by_age_group_month = pop_by_age_group_month.reindex(columns=AGE_GROUP_NAMES, fill_value=0).cumsum(axis=1)
        daily_hospitalization_counts = daily_hospitalization_counts.reindex(columns=AGE_GROUP_NAMES, fill_value=0).cumsum(axis=1)
    
    # save a three layered array with total counts, positive counts, and total hospitalizations for each date and age group
    if save_counts:
        # Create complete date range covering all test dates
        date_range = pd.date_range(start=min(positive_counts.index.min(), total_counts.index.min()), 
                      end=max(positive_counts.index.max(), total_counts.index.max()), 
                      freq='D')
        
        # Reindex to include all dates and fill with zeros
        positive_counts_filled = positive_counts.reindex(date_range, fill_value=0)
        total_counts_filled = total_counts.reindex(date_range, fill_value=0)
        daily_hospitalization_counts_filled = daily_hospitalization_counts.reindex(date_range, fill_value=0)

        # Censor the output data if mode is active (leaves cell blank/NaN in CSV)
        if censor:
            positive_counts_filled = positive_counts_filled.mask((positive_counts_filled > 0) & (positive_counts_filled < 5))
            total_counts_filled = total_counts_filled.mask((total_counts_filled > 0) & (total_counts_filled < 5))
            daily_hospitalization_counts_filled = daily_hospitalization_counts_filled.mask((daily_hospitalization_counts_filled > 0) & (daily_hospitalization_counts_filled < 5))

        # find number of age groups
        NAG = len(AGE_GROUP_NAMES)

        # Save as separate CSV files with no index or column names
        positive_counts_filled.to_csv(f'Data/Processed/KPSC_panel_{pathogen}_positive_counts'+['', '_hospday'][hosp]+['', '_split'][NAG>7]+["", "_sac"][sac]+['', '_dedup'][dedup]+['', '_cumulative'][cum_sum]+'.csv', header=False, index=False)
        total_counts_filled.to_csv(f'Data/Processed/KPSC_panel_{pathogen}_total_counts'+['', '_hospday'][hosp]+['', '_split'][NAG>7]+["", "_sac"][sac]+['', '_dedup'][dedup]+['', '_cumulative'][cum_sum]+'.csv', header=False, index=False)
        daily_hospitalization_counts_filled.to_csv(f'Data/Processed/KPSC_panel_hospitalizations_noCOVID'+['', '_split'][NAG>7]+["", "_sac"][sac]+['', '_dedup'][dedup]+['', '_cumulative'][cum_sum]+'.csv', header=False, index=False)

    if aggregation == "D":
        if censor:
            positive_counts = positive_counts.mask((positive_counts > 0) & (positive_counts < 5))
            total_counts = total_counts.mask((total_counts > 0) & (total_counts < 5))
            daily_hospitalization_counts = daily_hospitalization_counts.mask((daily_hospitalization_counts > 0) & (daily_hospitalization_counts < 5))

        # When both positive and total counts are zero, set proportion to 0
        prop_pos = positive_counts / total_counts.replace(0, np.nan)
        prop_pos = prop_pos.fillna(0)  # Fill NaN values (from 0/0 or censoring) with 0

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
        
        if return_ci:
            # Effective sample size over the smoothing window
            n_eff = total_counts.rolling(window=window_size, center=True, min_periods=1).sum().replace(0, np.nan)
            z = 1.96
            denom = 1 + (z**2) / n_eff
            center = (prop_pos_smoothed + (z**2) / (2 * n_eff)) / denom
            spread = (z * np.sqrt((prop_pos_smoothed * (1 - prop_pos_smoothed)) / n_eff + (z**2) / (4 * (n_eff**2)))) / denom
            p_lower = center - spread
            p_upper = center + spread
            
            # Underlying Wald variance kept strictly for macro-level age group aggregation steps
            var_p = (prop_pos_smoothed * (1 - prop_pos_smoothed)) / n_eff
            var_p = var_p.fillna(0)

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
                
                result = prop_pos_smoothed_summed.to_frame(name='Total')
                if return_ci:
                    n_eff_sum = total_counts_summed.rolling(window=window_size, center=True, min_periods=1).sum().replace(0, np.nan)
                    denom_sum = 1 + (z**2) / n_eff_sum
                    center_sum = (prop_pos_smoothed_summed + (z**2) / (2 * n_eff_sum)) / denom_sum
                    spread_sum = (z * np.sqrt((prop_pos_smoothed_summed * (1 - prop_pos_smoothed_summed)) / n_eff_sum + (z**2) / (4 * (n_eff_sum**2)))) / denom_sum
                    lower = center_sum - spread_sum
                    upper = center_sum + spread_sum
                    return result, lower.to_frame(name='Total').fillna(0), upper.to_frame(name='Total').fillna(0)
                return result
            else:
                if return_ci:
                    return prop_pos_smoothed, p_lower.fillna(0), p_upper.fillna(0)
                return prop_pos_smoothed
            
        if return_counts:
            aligned_hosp_counts = daily_hospitalization_counts.reindex(prop_pos_smoothed.index).fillna(0)
            incidence = prop_pos_smoothed * aligned_hosp_counts
            if return_ci:
                var_inc = (aligned_hosp_counts ** 2) * var_p
                lower = p_lower * aligned_hosp_counts
                upper = p_upper * aligned_hosp_counts
        else:
            # Align with hospitalization rates and calculate incidence
            pop_by_age_group_daily = pop_by_age_group_month.resample('D').ffill()
            pop_by_age_group_daily = pop_by_age_group_daily.reindex(columns=AGE_GROUP_NAMES, fill_value=0)
            daily_hospitalization_rates = daily_hospitalization_counts.div(pop_by_age_group_daily, axis=1)
            aligned_hosp_rates = daily_hospitalization_rates.reindex(prop_pos_smoothed.index)
            incidence = prop_pos_smoothed * aligned_hosp_rates
            if return_ci:
                var_inc = (aligned_hosp_rates ** 2) * var_p
                lower = p_lower * aligned_hosp_rates
                upper = p_upper * aligned_hosp_rates
            
        incidence = incidence.fillna(0)
        if return_ci:
            var_inc = var_inc.fillna(0)
            lower = lower.fillna(0)
            upper = upper.fillna(0)

    else:
        # Resample to desired aggregation
        positive_counts_agg = positive_counts.resample(aggregation).sum()
        total_counts_agg = total_counts.resample(aggregation).sum()

        if censor:
            positive_counts_agg = positive_counts_agg.mask((positive_counts_agg > 0) & (positive_counts_agg < 5))
            total_counts_agg = total_counts_agg.mask((total_counts_agg > 0) & (total_counts_agg < 5))

        # Calculate proportion positive
        prop_pos_agg = positive_counts_agg / total_counts_agg.replace(0, np.nan)
        prop_pos_agg = prop_pos_agg.fillna(0)  # Fill NaN values (from 0/0 or censoring) with 0
        
        if return_ci:
            n_safe = total_counts_agg.replace(0, np.nan)
            z = 1.96
            denom = 1 + (z**2) / n_safe
            center = (prop_pos_agg + (z**2) / (2 * n_safe)) / denom
            spread = (z * np.sqrt((prop_pos_agg * (1 - prop_pos_agg)) / n_safe + (z**2) / (4 * (n_safe**2)))) / denom
            p_lower = center - spread
            p_upper = center + spread
            
            var_p = (prop_pos_agg * (1 - prop_pos_agg)) / n_safe
            var_p = var_p.fillna(0)

        if pp_only:
            if sum_age_groups:
                # Sum positive and total counts across age groups
                positive_counts_summed = positive_counts_agg.sum(axis=1)
                total_counts_summed = total_counts_agg.sum(axis=1)
                prop_pos_summed = positive_counts_summed / total_counts_summed.replace(0, np.nan)
                prop_pos_summed = prop_pos_summed.fillna(0)
                
                result = prop_pos_summed.to_frame(name='Total')
                if return_ci:
                    n_eff_sum = total_counts_summed.replace(0, np.nan)
                    denom_sum = 1 + (z**2) / n_eff_sum
                    center_sum = (prop_pos_summed + (z**2) / (2 * n_eff_sum)) / denom_sum
                    spread_sum = (z * np.sqrt((prop_pos_summed * (1 - prop_pos_summed)) / n_eff_sum + (z**2) / (4 * (n_eff_sum**2)))) / denom_sum
                    lower = center_sum - spread_sum
                    upper = center_sum + spread_sum
                    return result, lower.to_frame(name='Total').fillna(0), upper.to_frame(name='Total').fillna(0)
                return result
            else:
                if return_ci:
                    return prop_pos_agg, p_lower.fillna(0), p_upper.fillna(0)
                return prop_pos_agg
                
        if return_counts:
            daily_hospitalization_counts_agg = daily_hospitalization_counts.resample(aggregation).sum()
            if censor:
                daily_hospitalization_counts_agg = daily_hospitalization_counts_agg.mask((daily_hospitalization_counts_agg > 0) & (daily_hospitalization_counts_agg < 5))
            aligned_hosp_counts_agg = daily_hospitalization_counts_agg.reindex(prop_pos_agg.index).fillna(0)
            incidence =  prop_pos_agg * aligned_hosp_counts_agg
            if return_ci:
                var_inc = (aligned_hosp_counts_agg ** 2) * var_p
                lower = p_lower * aligned_hosp_counts_agg
                upper = p_upper * aligned_hosp_counts_agg
        else:
            # Calculate incidence
            pop_by_age_group_agg = pop_by_age_group_month.resample(aggregation).ffill()
            pop_by_age_group_agg = pop_by_age_group_agg.reindex(columns=AGE_GROUP_NAMES, fill_value=0)
            daily_hospitalization_counts_agg = daily_hospitalization_counts.resample(aggregation).sum()
            if censor:
                daily_hospitalization_counts_agg = daily_hospitalization_counts_agg.mask((daily_hospitalization_counts_agg > 0) & (daily_hospitalization_counts_agg < 5))
            daily_hospitalization_rates_agg = daily_hospitalization_counts_agg.div(pop_by_age_group_agg, axis=1)
            aligned_hosp_rates_agg = daily_hospitalization_rates_agg.reindex(prop_pos_agg.index)
            incidence = prop_pos_agg * aligned_hosp_rates_agg
            if return_ci:
                var_inc = (aligned_hosp_rates_agg ** 2) * var_p
                lower = p_lower * aligned_hosp_rates_agg
                upper = p_upper * aligned_hosp_rates_agg
            
        incidence = incidence.fillna(0)
        if return_ci:
            var_inc = var_inc.fillna(0)
            lower = lower.fillna(0)
            upper = upper.fillna(0)
    
    # Sum age groups at the end if requested
    if sum_age_groups:
        if return_counts:
            result = incidence.sum(axis=1).to_frame(name='Total')
            if return_ci:
                var_inc_sum = var_inc.sum(axis=1).to_frame(name='Total')
                lower = np.maximum(0, result - 1.96 * np.sqrt(var_inc_sum))
                upper = result + 1.96 * np.sqrt(var_inc_sum)
                return result, lower, upper
            return result

        # Weight by population when summing
        if aggregation == "D":
            pop_weights = pop_by_age_group_daily.reindex(incidence.index)
        else:
            pop_weights = pop_by_age_group_agg.reindex(incidence.index)
        
        # Calculate weighted sum
        weighted_incidence = (incidence * pop_weights).sum(axis=1)
        total_population = pop_weights.sum(axis=1)
        result = (weighted_incidence / total_population).to_frame(name='Total')
        
        if return_ci:
            # Propagate variance through the weighted sum calculation (Safe via CLT on aggregated populations)
            var_inc_sum = ((var_inc * (pop_weights ** 2)).sum(axis=1) / (total_population ** 2)).to_frame(name='Total')
            lower = np.maximum(0, result - 1.96 * np.sqrt(var_inc_sum))
            upper = result + 1.96 * np.sqrt(var_inc_sum)
            return result, lower, upper
            
        return result
    
    if return_ci:
        return incidence, lower, upper
        
    return incidence

def load_and_filter_test_data(remove_salvage=True):
    # time_start = time.time()
    test_data1 = pd.read_sas('Data/Raw/KPSC/testing.sas7bdat', format='sas7bdat', encoding='utf-8')
    test_data2 = pd.read_sas('Data/Raw/KPSC/testing_20250818.sas7bdat', format='sas7bdat', encoding='utf-8')
    test_data = pd.concat([test_data1, test_data2], ignore_index=True)
    
    # Create date columns for efficient matching
    test_data.loc[:, 'Test date'] = (
        pd.to_datetime(test_data["YEAR"].astype(int).astype(str) + '-10-01') +
        pd.to_timedelta(test_data["lab_days"].astype(int), unit='D')
    ).dt.normalize()

    if remove_salvage:
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

        matched_salvage = merged['_merge'].eq('both').sum()
        print(f"RSV salvage data points matched: {matched_salvage} out of {len(rsv_salvage_renamed)}")
        
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

    pathogen_list = ["InfluenzaA","InfluenzaB","RSV","Metapneumovirus","Adenovirus","Parainfluenza3","Enterovirus"]
    # Keep only panel tests, i.e. there is a test for each pathogen in the list (that is, containing the string for each pathogen in the list in the pathogen column) for a given StudyID and test date
    panel_test_groups = test_data.groupby(["StudyID", date_name])["pathogen"].apply(lambda x: all(any(p == pathogen for pathogen in x) for p in pathogen_list))
    test_data = test_data.set_index(["StudyID", date_name]).loc[panel_test_groups[panel_test_groups].index].reset_index()
    return test_data

def load_and_filter_clinical_data(exclude_covid=True, settings=["Hospital admission"]):
    # time_start = time.time()
    clinical_data1 = pd.read_sas('Data/Raw/KPSC/clinical_20241202.sas7bdat', format='sas7bdat', encoding='utf-8')
    clinical_data2 = pd.read_sas('Data/Raw/KPSC/clinical_20260203.sas7bdat', format='sas7bdat', encoding='utf-8')
    clinical_data = pd.concat([clinical_data1, clinical_data2], ignore_index=True)
    # print("Time to load clinical data: ",time.time()-time_start)

    clinical_data = clinical_data[clinical_data["dxgroup"] == "ARI"]

    clinical_data = clinical_data[clinical_data["setting"].isin(settings)]

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

def filter_multiple_hospitalizations(clinical_data):
    # Keep only one hospitalization per StudyID, prioritizing earliest date
    clinical_data = clinical_data.sort_values(by=["StudyID", "Hospitalization date"], ascending=[True, True])
    clinical_data.loc[:, "diff"] = clinical_data.groupby("StudyID")["Hospitalization date"].diff().dt.days
    clinical_data = clinical_data[(clinical_data["diff"].isna()) | (clinical_data["diff"] > 14)].drop(columns=["diff"])
    return clinical_data

def bin_age_groups(data, AGE_GROUPS, AGE_GROUP_NAMES):
    bins = [group[0] for group in AGE_GROUPS] + [AGE_GROUPS[-1][-1] + 1]
    data.loc[:,"age_group"] = pd.cut(data["age_in_mo"],bins=bins,labels=AGE_GROUP_NAMES,right=False)
    return data

def merge_tests(test_data, clinical_data, clinical_date_name="Hospitalization date", test_date_name="Test date"):
    test_data = test_data[test_data["StudyID"].isin(clinical_data["StudyID"])].copy()
    # # find date of from clinical_data for each study ID and match to tests
    test_data = clinical_data.merge(test_data[["StudyID",test_date_name,"pathogen","lab_type","lab_days","result_val"]],on="StudyID",how="left")
    # # # keep only rows where Hospitalization date is within 14 days of Test date
    test_data = test_data[np.abs((pd.to_datetime(test_data[clinical_date_name]) - pd.to_datetime(test_data[test_date_name])).dt.days) <= 14]
    return test_data

def filter_multiple_testing(test_data):
    # Keep only one test per pathogen and hospitalization - prioritize positive tests
    test_data = test_data.sort_values(by=["StudyID","pathogen","result_val","Test date"], ascending=[True,True,False,True])
    test_data.loc[:,"diff"] = test_data.groupby(["StudyID","pathogen"])["Test date"].diff().dt.days
    test_data = test_data[(test_data["diff"].isna()) | (test_data["diff"] > 14)]
    return test_data.drop(columns=["diff"])

def get_daily_test_counts(test_data, assigned_date="Hospitalization date"):
    # Create daily aggregations for total tests and positive tests
    daily_total_counts = test_data.groupby([assigned_date, 'pathogen', 'age_group']).size().reset_index(name='count')
    daily_total_counts.loc[:,'result_type'] = 'Total'

    daily_positive_counts = test_data[test_data['result_val'] == 'Positive'].groupby([assigned_date, 'pathogen', 'age_group']).size().reset_index(name='count')
    daily_positive_counts.loc[:,'result_type'] = 'Positive'

    # Combine total and positive counts
    daily_test_counts = pd.concat([daily_total_counts, daily_positive_counts], ignore_index=True)

    # Create complete date range and pathogen/result combinations
    date_range = pd.date_range(start=daily_test_counts[assigned_date].min(), 
                            end=daily_test_counts[assigned_date].max(), 
                            freq='D')
    all_pathogens = daily_test_counts['pathogen'].unique()
    all_age_groups = daily_test_counts['age_group'].unique()
    result_types = ['Total', 'Positive']

    # Create complete combinations
    complete_combinations = pd.MultiIndex.from_product([date_range, all_pathogens, all_age_groups, result_types], 
                                                    names=[assigned_date, 'pathogen', 'age_group', 'result_type']).to_frame(index=False)

    # Merge with actual counts and fill zeros
    daily_test_counts_complete = complete_combinations.merge(daily_test_counts, 
                                                            on=[assigned_date, 'pathogen', 'age_group', 'result_type'], 
                                                            how='left')
    daily_test_counts_complete.loc[:,'count'] = daily_test_counts_complete['count'].fillna(0).astype(int)
    return daily_test_counts_complete



if __name__ == "__main__" and False:
    from Parameters.census_population import AGE_GROUP_NAMES_sac as AGE_GROUP_NAMES
    NAG = len(AGE_GROUP_NAMES)
    hsv_colors = colormaps.hsv(-0.02+np.arange(NAG)/NAG)
    hsv_colors[3] = colormaps.hsv((3/NAG)+0.28/NAG)
    fig, ax = plt.subplots(3, 2, figsize=(13.3, 7.5), sharex=True)
    
    for pi, pathogen in enumerate(["RSV", "Metapneumovirus", "Parainfluenza3", "Adenovirus", "InfluenzaA", "InfluenzaB"]):
        incidence = calculate_proportion_positive_incidence(pathogen, aggregation="MS", window_size=1, weighting_factor=0, sum_age_groups=False, save_counts=False, pp_only=False, hosp=True, NAG=NAG, dedup=True, sac=True, censor=False)        
        for i, age_group in enumerate(AGE_GROUP_NAMES):
            ax[pi // 2, pi % 2].plot(incidence.index, incidence[age_group] * 100000, color=hsv_colors[i], label=AGE_GROUP_NAMES[i], linewidth=1.5)
        # ax[pi // 2, pi % 2].plot(incidence.index, incidence['Total'] * 100000, color='black', label='Total', linewidth=2)
        ax[pi // 2, pi % 2].set_title(f"{pathogen}")
        ax[pi // 2, pi % 2].set_ylabel('Incidence per 100k')
        # print(pathogen, incidence['Total'].max() * 100000 / 25)
        # ax[pi // 2, pi % 2].axhline(incidence['Total'].max() * 100000 / 25, color='gray', linestyle='--', linewidth=1)
    
    # ax[3, 1].axis('off')
    # ax[3, 1].legend(handles=[Rectangle((0, 0), 1, 1, color=hsv_colors[i]) for i in range(NAG)], labels=AGE_GROUP_NAMES, loc='center')
    
    plt.tight_layout()
    plt.savefig("Figures/KPSC_panel_all_pathogens_incidence_monthly_sac_dedup_age.png", dpi=300)


#### 
if __name__ == "__main__":
    from Parameters.census_population import AGE_GROUPS_sac as AGE_GROUPS, AGE_GROUP_NAMES_sac as AGE_GROUP_NAMES
    AGE_GROUPS = [
        np.arange(3),
        np.arange(3,12),
        np.arange(12,5*12),
        np.arange(5*12,18*12),
        np.arange(18*12,40*12),
        np.arange(40*12,65*12),
        np.arange(65*12,100*12)]
    time_start = time.time()
    test_data = load_and_filter_test_data(remove_salvage=True)
    time_test = time.time()
    print(f"Time to load test data: {time_test - time_start:.2f}s")

    # print("Skipping filtering to panel tests to compare all viruses")
    # if False:
    #     test_data = filter_to_panel_tests(test_data)
    #     time_panel = time.time()
    #     print(f"Time to filter to panel tests: {time_panel - time_test:.2f}s")
    # else:
    #     time_panel = time_test

    # hospitalization_data = load_and_filter_clinical_data(exclude_covid=True, settings=["Hospital admission"])
    # time_hosp = time.time()
    # # # print(f"Time to load and filter all_clinical data: {time_hosp - time_test:.2f}s")
    # # # remove multiple hospitalizations within 14 days
    # hospitalization_data_filtered = filter_multiple_hospitalizations(hospitalization_data)
    # print(hospitalization_data_filtered["race_eth_c"].value_counts())
    # # bin NDI into <-1, 1-0, 0-1, >1
    # hospitalization_data_filtered.loc[:,"NDI_bin"] = pd.cut(hospitalization_data_filtered["NDI"], bins=[-np.inf, -1, 0, 1, np.inf], labels=["<-1", "-1-0", "0-1", ">1"])
    # print(hospitalization_data_filtered["NDI_bin"].value_counts())
    # hospitalization_data_filtered = bin_age_groups(hospitalization_data_filtered, AGE_GROUPS, AGE_GROUP_NAMES)
    # print(hospitalization_data_filtered["age_group"].value_counts())
    # print(hospitalization_data_filtered["YEAR"].value_counts())
    # print(hospitalization_data_filtered["StudyID"].nunique())
    # # Set all_clinical date as index for resampling
    # daily_hospitalization_counts = hospitalization_data_filtered.pivot_table(index='Hospitalization date', columns='age_group', values='StudyID', aggfunc='count').fillna(0).reset_index()
    # # # reorder columns
    # daily_hospitalization_counts = daily_hospitalization_counts[['Hospitalization date'] + AGE_GROUP_NAMES]
    # # set index to all_clinical date
    # daily_hospitalization_counts = daily_hospitalization_counts.set_index('Hospitalization date')
    # print(f"Time to process hospitalization data: {time.time() - time_hosp:.2f}s")

    # test_data = merge_tests(test_data, hospitalization_data_filtered)
    # test_data = filter_multiple_testing(test_data)
    # print(test_data.columns)
    # daily_test_counts = get_daily_test_counts(test_data)
    # print(f"Time to process test data: {time.time() - time_panel:.2f}s")

    # # print(test_data["result_val"].value_counts())
    # print("Pathogen counts for test-positive ARI hospitalizations:")
    # print(test_data[(test_data["result_val"] == "Positive")]["pathogen"].value_counts())

    # # find total number of test-positive ARI hospitalizations with any non-covid pathogen, and total number of test-positive hospitalizations with the six focal pathogens
    # test_positive_non_covid = test_data[(test_data["result_val"] == "Positive") & (test_data["pathogen"] != "SARS-COV-2 (COVID-19)")]
    # test_positive_focal = test_data[(test_data["result_val"] == "Positive") & (test_data["pathogen"].isin(["INFLUENZA VIRUS B", "INFLUENZA VIRUS A", "RESPIRATORY SYNCYTIAL VIRUS SUBTYPE B", "RESPIRATORY SYNCYTIAL VIRUS SUBTYPE A", "RESPIRATORY SYNCYTIAL VIRUS", "INFLUENZA A VIRUS", "INFLUENZA VIRUS", "INFLUENZA A VIRUS SUBTYPE H1", "INFLUENZA A H1N1 2009", "INFLUENZA A VIRUS SUBTYPE/HEMAGGLUTININ H3", "HUMAN METAPNEUMOVIRUS VIRUS", "ADENOVIRUS", "PARAINFLUENZA VIRUS 3"]))]
    # print(f"Total test-positive non-COVID ARI hospitalizations: {test_positive_non_covid['StudyID'].nunique()}")
    # print(f"Total test-positive focal pathogen ARI hospitalizations: {test_positive_focal['StudyID'].nunique()}")
    # print(f"Focal pathogens as proportion of all non-COVID ARI hospitalizations: {test_positive_focal['StudyID'].nunique() / test_positive_non_covid['StudyID'].nunique():.2%}")

    # # load and split the 8-39 age group pop_by_age_group_monthly into 8-17 and 18-39 age groups assuming a (17-8)/(39-8) proportion
    # pop_by_age_group_month = pd.read_csv('Data/Processed/KPSC_population_by_age_group_monthly.csv', index_col=0, parse_dates=['month_start'])
    # pop_by_age_group_month['5-17y'] = pop_by_age_group_month['5-7y'] + pop_by_age_group_month['8-39y'] * (18-8) / (40-8)
    # pop_by_age_group_month['18-39y'] = pop_by_age_group_month['8-39y'] * (40-18) / (40-8)
    # pop_by_age_group_month = pop_by_age_group_month.drop(columns=['5-7y'])
    # pop_by_age_group_month = pop_by_age_group_month.drop(columns=['8-39y'])
    # pop_by_age_group_month = pop_by_age_group_month.reindex(columns=AGE_GROUP_NAMES)

    # # # save daily hospitalization counts and pop by age group month for use in incidence calculation
    # daily_hospitalization_counts.to_csv(f'Data/Processed/KPSC_ARI_nonCOVID_hospitalizations_by_day_age_group_sac_dedup_allvirus.csv')
    # pop_by_age_group_month.to_csv(f'Data/Processed/KPSC_population_by_age_group_monthly_sac_allvirus.csv')

    # # # load counts and pop by age group month from csv files
    # # daily_hospitalization_counts = pd.read_csv(f'Data/Processed/KPSC_ARI_nonCOVID_hospitalizations_by_day_age_group.csv', index_col=0, parse_dates=True)
    # # pop_by_age_group_month = pd.read_csv(f'Data/Processed/KPSC_population_by_age_group_monthly.csv', index_col=0, parse_dates=['month_start'])
    # # # # save daily hospitalization rates by age group for use in incidence calculation
    # daily_hospitalization_rates = daily_hospitalization_counts.div(pop_by_age_group_month.resample('D').ffill(), axis=1)
    # daily_hospitalization_rates.to_csv(f'Data/Processed/KPSC_ARI_nonCOVID_hospitalization_rates_by_day_age_group_sac_dedup_allvirus.csv')
    # # # # save daily test counts for use in incidence calculation
    # daily_test_counts = daily_test_counts.set_index('Hospitalization date')
    # daily_test_counts.to_csv(f'Data/Processed/KPSC_ARI_hospitalized_pathogen_panel_test_counts_by_hosp_day_pathogen_age_group_sac_dedup_allvirus.csv')

    # NAG = len(AGE_GROUP_NAMES)
    # hsv_colors = colormaps.hsv(-0.02+np.arange(NAG)/NAG)
    # hsv_colors[3] = colormaps.hsv((3/NAG)+0.28/NAG)


    # # load daily test counts
    # daily_test_counts = pd.read_csv(f'Data/Processed/KPSC_ARI_hospitalized_pathogen_panel_test_counts_by_hosp_day_pathogen_age_group_sac_dedup.csv', index_col=0, parse_dates=True)
    # # for each pathogen, plot proportion positive and total tests over time on the same plot for each age group
    # pathogens = daily_test_counts['pathogen'].unique()
    # for pathogen in pathogens:
    #     fig, ax = plt.subplots(4, 2, figsize=(13.3, 7.5), sharex=True)
    #     for i, age_group in enumerate(AGE_GROUP_NAMES):
    #         age_group_data = daily_test_counts[(daily_test_counts['pathogen'] == pathogen) & (daily_test_counts['age_group'] == age_group)]
    #         age_group_data = age_group_data.sort_index()
    #         total_tests = age_group_data[age_group_data['result_type'] == 'Total']['count'].resample('MS').sum()
    #         positive_tests = age_group_data[age_group_data['result_type'] == 'Positive']['count'].resample('MS').sum()
    #         prop_positive = positive_tests / total_tests.replace(0, np.nan)
    #         prop_positive = prop_positive.fillna(0)
    #         ax[i//2, i%2].plot(prop_positive.index, prop_positive.values, color=hsv_colors[i], label='Proportion Positive', linewidth=1.5)
    #         ax[i//2, i%2].plot(total_tests.index, total_tests.values / total_tests.max(), color='gray', label='Total Tests (scaled)')
    #         ax[i//2, i%2].set_title(f"{pathogen} - {age_group}")
        
    #     ax[1, 0].set_ylabel('Proportion Positive / Total Tests (scaled)')
    #     plt.tight_layout()
    #     plt.savefig(f"Figures/KPSC_panel_{pathogen}_proportion_positive_and_total_tests_by_age_group_monthly_sac_dedup.png", dpi=300)

    # daily_hospitalization_rates = daily_hospitalization_counts.div(pop_by_age_group_month.resample('D').ffill(), axis=1)
    # daily_hospitalization_rates.to_csv(f'Data/Processed/KPSC_ARI_nonCOVID_clinical_rates_by_day_age_group_split.csv')

    # # load daily hospitalization rates and plot by age group in a 4 x 2 panel plot (data before 2020 only)
    # daily_hospitalization_rates = pd.read_csv(f'Data/Processed/KPSC_ARI_hospitalization_rates_by_day_age_group_split.csv', index_col=0)
    # daily_hospitalization_rates.index = pd.to_datetime(daily_hospitalization_rates.index)
    
    # breakpoint1 = pd.to_datetime('2020-03-19')
    # breakpoint2 = pd.to_datetime('2022-03-01')
    # endpoint = pd.to_datetime('2025-05-01')
    
    # # Plot 1: Individual age group panels
    # fig, ax = plt.subplots(4,2,figsize=(13.3,7.5),sharex=True)
    
    # for i,age_group in enumerate(AGE_GROUP_NAMES):
    #     age_group_data = daily_hospitalization_rates[age_group]
    #     age_group_data_weekly = age_group_data.resample('W').sum()
    #     ax[i//2, i%2].plot(age_group_data_weekly.index, age_group_data_weekly.values * 100000, color="k", label=age_group)
        
    #     # Fit before and after breakpoint
    #     before_breakpoint = age_group_data_weekly[age_group_data_weekly.index < breakpoint1]
    #     after_breakpoint = age_group_data_weekly[(age_group_data_weekly.index >= breakpoint2) & (age_group_data_weekly.index < endpoint)]
        
    #     # Fit line before breakpoint
    #     non_zero_before = before_breakpoint[before_breakpoint > 0]
    #     if len(non_zero_before) > 1:
    #         slope_before, intercept_before, r_value_before, p_value_before, std_err_before = stats.linregress(non_zero_before.index.astype(int), non_zero_before.values)
    #         ax[i//2, i%2].plot(non_zero_before.index, 100000*(slope_before*non_zero_before.index.astype(int) + intercept_before), color="r", linestyle="--", label="Trend (before)")
        
    #     # Fit line after breakpoint
    #     non_zero_after = after_breakpoint[after_breakpoint > 0]
    #     if len(non_zero_after) > 1:
    #         slope_after, intercept_after, r_value_after, p_value_after, std_err_after = stats.linregress(non_zero_after.index.astype(int), non_zero_after.values)
    #         ax[i//2, i%2].plot(non_zero_after.index, 100000*(slope_after*non_zero_after.index.astype(int) + intercept_after), color="orange", linestyle="--", label="Trend (after)")
        
    #     # Display trend info
    #     if len(non_zero_before) > 1 and len(non_zero_after) > 1:
    #         trend_text = f"Before: slope={slope_before:.2e}, p={p_value_before:.2f}\nAfter: slope={slope_after:.2e}, p={p_value_after:.2f}"
    #     elif len(non_zero_before) > 1:
    #         trend_text = f"Before: slope={slope_before:.2e}, p={p_value_before:.2f}"
    #     elif len(non_zero_after) > 1:
    #         trend_text = f"After: slope={slope_after:.2e}, p={p_value_after:.2f}"
    #     else:
    #         trend_text = ""
        
    #     if trend_text:
    #         ax[i//2, i%2].text(0.98, 0.97, trend_text, transform=ax[i//2, i%2].transAxes, 
    #                   verticalalignment='top', horizontalalignment='right', 
    #                   fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    #     ax[i//2, i%2].axvline(breakpoint1, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    #     ax[i//2, i%2].axvline(breakpoint2, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    #     ax[i//2, i%2].axvline(endpoint, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    #     ax[i//2, i%2].set_title(f"{age_group}")
    
    # ax[1, 0].set_ylabel('Hospitalization rate per 100k')
    # plt.tight_layout()
    # plt.savefig(f"Figures/KPSC_panel_hospitalization_rates_by_age_group_weekly_endpoint.png",dpi=300)
   
    # # Detrend only the post-breakpoint period (>= breakpoint2) when post-breakpoint trend is significant
    # fig, ax = plt.subplots(4,2,figsize=(13.3,7.5),sharex=True)
    
    # # Initialize DataFrame to store detrended data
    # detrended_rates = daily_hospitalization_rates.copy()

    # for i, age_group in enumerate(AGE_GROUP_NAMES):
    #     age_group_data = daily_hospitalization_rates[age_group]
    #     age_group_data_daily = age_group_data.resample('D').sum()

    #     # Start from raw weekly rate per 100k
    #     plot_series = age_group_data_daily * 100000.0
    #     title_suffix = ""

    #     # Fit trend using only post-breakpoint window
    #     post_mask = (age_group_data_daily.index >= breakpoint2) & (age_group_data_daily.index < endpoint)
    #     post_data = age_group_data_daily.loc[post_mask]
    #     non_zero_post = post_data[post_data > 0]

    #     if len(non_zero_post) > 1:
    #         x_post = non_zero_post.index.astype(np.int64)
    #         slope_after, intercept_after, r_value_after, p_value_after, std_err_after = stats.linregress(
    #             x_post, non_zero_post.values
    #         )

    #         # Detrend only post-breakpoint dates if significant
    #         if (p_value_after < 0.01) and (slope_after > 0):
    #             print(f"Detrending post-breakpoint period for {age_group} due to significant trend (p={p_value_after:.4f})")
    #             x_all_post = age_group_data_daily.loc[post_mask].index.astype(np.int64)
    #             fitted_post = pd.Series(
    #                 slope_after * x_all_post + intercept_after,
    #                 index=age_group_data_daily.loc[post_mask].index
    #             )
    #             print(intercept_after, slope_after, p_value_after)
    #             valid = fitted_post > 0
    #             detrended_post = age_group_data_daily.loc[post_mask].copy()
    #             detrended_post.loc[valid] = (detrended_post.loc[valid] - fitted_post.loc[valid]) + intercept_after + slope_after * x_all_post[0]
    #             # minimum of zero
    #             detrended_post.loc[detrended_post < 0] = 0
    #             plot_series.loc[post_mask] = detrended_post * 100000
    #             title_suffix = " (post detrended)"
                
    #             # Update daily detrended rates for post-breakpoint period
    #             post_daily_mask = (detrended_rates.index >= breakpoint2) & (detrended_rates.index < endpoint)
    #             fitted_post_aligned = fitted_post.reindex(detrended_rates.loc[post_daily_mask].index, method='nearest').fillna(0)
    #             detrended_post_values = detrended_rates.loc[post_daily_mask, age_group].values - fitted_post_aligned + intercept_after + slope_after * x_all_post[0]
    #             # Ensure no negative values
    #             detrended_post_values = np.maximum(detrended_post_values, 0)
    #             detrended_rates.loc[post_daily_mask, age_group] = detrended_post_values

    #     # Plot pre-breakpoint2 data in one color, post-breakpoint2 in another
    #     pre_mask = age_group_data_daily.index < breakpoint2
    #     if pre_mask.any():
    #         ax[i//2, i%2].plot(plot_series.loc[pre_mask].index, plot_series.loc[pre_mask].values, 
    #                           color="k", label=age_group, linewidth=1.5)
        
    #     if post_mask.any():
    #         # plot undetrended post-breakpoint data in light gray for reference
    #         ax[i//2, i%2].plot(age_group_data_daily.loc[post_mask].index, age_group_data_daily.loc[post_mask].values * 100000, 
    #                           color="lightgray", label=f"{age_group} (undetrended)", linewidth=1.5, linestyle='--')
    #         # plot detrended post-breakpoint data
    #         ax[i//2, i%2].plot(plot_series.loc[post_mask].index, plot_series.loc[post_mask].values, 
    #                           color="steelblue", label=f"{age_group} (detrended)" if title_suffix else age_group, 
    #                           linewidth=1.5)
        
    #     ax[i//2, i%2].set_title(f"{age_group}{title_suffix}")
    #     ax[i//2, i%2].axvline(breakpoint1, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    #     ax[i//2, i%2].axvline(breakpoint2, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    #     ax[i//2, i%2].axvline(endpoint, color='gray', linestyle=':', alpha=0.7, linewidth=1)

    # ax[1, 0].set_ylabel('Hospitalization rate per 100k (post-breakpoint detrended if p<0.01)')
    # plt.tight_layout()
    # plt.savefig("Figures/KPSC_panel_hospitalization_rates_by_age_group_daily_detrended.png", dpi=300)
    
    # # Save detrended rates in same format as original
    # detrended_rates.to_csv('Data/Processed/KPSC_ARI_nonCOVID_hospitalization_rates_by_day_age_group_split_detrended.csv')

    # # multiply by population to get detrended hospitalization counts and save
    # detrended_counts = detrended_rates.mul(pop_by_age_group_month.resample('D').ffill(), axis=1)
    # detrended_counts.to_csv('Data/Processed/KPSC_ARI_nonCOVID_hospitalizations_by_day_age_group_split_detrended.csv')

    # # load daily test counts for use in incidence calculation
    # daily_test_counts = pd.read_csv(f'Data/Processed/KPSC_ARI_hospitalized_pathogen_panel_test_counts_by_hosp_day_pathogen_age_group_split.csv', index_col=0, parse_dates=['Hospitalization date'])

    # # plot total number of tests in hospitalized patients by week in each age group ina 4 x 2 panel plot
    # fig, ax = plt.subplots(4,2,figsize=(13.3,7.5),sharex=True)
    # for i,age_group in enumerate(AGE_GROUP_NAMES):
    #     age_group_data = daily_test_counts[daily_test_counts['age_group'] == age_group].groupby('Hospitalization date')['count'].sum()
    #     age_group_data.index = pd.to_datetime(age_group_data.index)
    #     age_group_data_weekly = age_group_data.resample('W').sum()
    #     ax[i//2, i%2].plot(age_group_data_weekly.index, age_group_data_weekly.values, color="k", label=age_group)
    #     ax[i//2, i%2].set_title(f"{age_group}")
    # ax[1, 0].set_ylabel('Number of tests')
    # plt.tight_layout()
    # plt.savefig(f"Figures/KPSC_panel_hosp_tests_by_age_group_weekly.png",dpi=300)

    # # for each pathogen plot incidence by age group in an eight panel plot
    # for pathogen in ["InfluenzaA","InfluenzaB","RSV","Metapneumovirus","Adenovirus","Parainfluenza3"]:
    #     incidence = calculate_proportion_positive_incidence(pathogen, aggregation="D", window_size=1, weighting_factor=0, sum_age_groups=False, save_counts=True, pp_only=False, hosp=True, pop_by_age_group_month=pop_by_age_group_month, daily_hospitalization_counts=daily_hospitalization_counts, daily_test_counts_complete=daily_test_counts)
    #     # fig, ax = plt.subplots(4,2,figsize=(13.3,7.5),sharex=True)
    #     # for i,age_group in enumerate(AGE_GROUP_NAMES):
    #     #     ax[i%4, i//4].plot(incidence.index, incidence[age_group] * 100000, color="k", label=age_group)
    #     #     ax[i%4, i//4].set_title(f"{age_group}")
    #     # ax[1, 0].set_ylabel('Incidence per 100k')
    #     # # ax[3,1].axis('off')
    #     # plt.tight_layout()
    #     # plt.savefig(f"Figures/{pathogen}_split_age_group_panels_hospitalization_incidence_daily.png",dpi=300)


    # combined_data = merge_positive_tests(test_data, hospitalization_data)
    # time_merge = time.time()
    # print(f"Time to merge positive tests with hospitalization data: {time_merge - time_hosp:.2f}s")

    # combined_data.to_csv('Data/Processed/KPSC_unsalvage_panel_positive_matched_noncovid_ARI_hospitalizations.csv',index=False)

    # combined_data = pd.read_csv('Data/Processed/KPSC_unsalvage_panel_positive_matched_noncovid_ARI_hospitalizations.csv', parse_dates=["Hospitalization date"])
    # # for each pathogen, aggregate the number of positive tests by age group and day
    # for pathogen in ["InfluenzaA","InfluenzaB","RSV","Metapneumovirus","Adenovirus","Parainfluenza3"]:
    #     pathogen_data = combined_data[combined_data["pathogen"].str.contains(pathogen,na=False)].copy()
    #     pathogen_data = pathogen_data.groupby(["Hospitalization date","AGE_GROUP"], observed=True)["StudyID"].count().reset_index()
    #     pathogen_data.rename(columns={"StudyID":"Positive tests"},inplace=True)
    #     # wide format with age groups as columns
    #     pathogen_data = pathogen_data.pivot(index="Hospitalization date", columns="AGE_GROUP", values="Positive tests").fillna(0).reset_index()
    #     # set order of age group columns
    #     pathogen_data = pathogen_data[["Hospitalization date"] + AGE_GROUP_NAMES]
    #     # fill dates from 2015-10-01 to 2025-05-01 with 0 positive tests for each age group
    #     all_dates = pd.date_range(start="2015-10-01", end="2025-05-01")
    #     pathogen_data = pathogen_data.set_index("Hospitalization date").reindex(all_dates).fillna(0).rename_axis("Hospitalization date").reset_index()
    #     pathogen_data.to_csv(f'Data/Processed/KPSC_unsalvage_panel_positive_{pathogen}_matched_noncovid_ARI_hospitalizations.csv',index=False)
    #     pop_by_age_group_month = pd.read_csv('Data/Processed/KPSC_population_by_age_group_monthly.csv', index_col=0, parse_dates=['month_start'])
    #     pop_by_age_group_month = pop_by_age_group_month.reindex(columns=AGE_GROUP_NAMES)
    #     pop_by_age_group_daily = pop_by_age_group_month.resample('D').ffill()
    #     # get proportional incidence by dividing positive tests by population
    #     pathogen_data[AGE_GROUP_NAMES] = pathogen_data[AGE_GROUP_NAMES].div(
    #         pop_by_age_group_daily.reindex(pathogen_data["Hospitalization date"]).values, axis=0
    #     )
    #     pathogen_data.to_csv(f'Data/Processed/KPSC_unsalvage_panel_positive_{pathogen}_matched_noncovid_ARI_hospitalizations_proportional_incidence.csv',index=False)
    


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