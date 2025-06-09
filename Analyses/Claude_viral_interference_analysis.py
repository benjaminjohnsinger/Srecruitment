import pandas as pd
import jax.numpy as np
import pyreadstat
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm
from statsmodels.formula.api import logit

# Load SAS data file
def load_sas_data(file_path):
    df, meta = pyreadstat.read_sas7bdat(file_path)
    return df

# Function to categorize pathogens into groups
def categorize_pathogens(pathogen):
    pathogen = str(pathogen).upper()
    
    if any(p in pathogen for p in ["INFLUENZA A", "INFLUENZA B", "INFLUENZA VIRUS"]):
        return "Influenza"
    elif any(p in pathogen for p in ["RESPIRATORY SYNCYTIAL VIRUS"]):
        return "RSV"
    elif "METAPNEUMOVIRUS" in pathogen:
        return "HMPV"
    elif "ADENOVIRUS" in pathogen:
        return "Adenovirus"
    elif "PARAINFLUENZA" in pathogen:
        return "Parainfluenza"
    else:
        return "Other"

# Process the data
def process_data(df):
    # Filter for PCR tests only
    df = df[df['lab_type'] == 'PCR'].copy()
    
    # Filter for valid results only
    df = df[df['result_val'] != 'Invalid'].copy()
    
    # Create binary result column (1 for positive, 0 for negative)
    df['is_positive'] = df['result_val'].apply(lambda x: 1 if x == 'Positive' else 0)
    
    # Group pathogens into categories
    df['pathogen_group'] = df['pathogen'].apply(categorize_pathogens)
    
    # Filter for only the pathogens of interest
    pathogens_of_interest = ['Influenza', 'RSV', 'HMPV', 'Adenovirus', 'Parainfluenza']
    df = df[df['pathogen_group'].isin(pathogens_of_interest)].copy()
    
    # Create more granular time variables
    # Week within year (more precise than season)
    df['week_within_year'] = df['lab_days'] // 7
    
    # Month within year
    df['month_within_year'] = df['lab_days'] // 30  # Approximate months
    
    # Create year-specific time variables
    # These distinguish the same time periods across different years
    df['year_week'] = df['YEAR'].astype(str) + '_' + df['week_within_year'].astype(str)
    df['year_month'] = df['YEAR'].astype(str) + '_' + df['month_within_year'].astype(str)
    
    # Keep original season for comparison
    df['season'] = pd.cut(
        df['lab_days'], 
        bins=[0, 90, 180, 270, 366],  # Roughly quarterly
        labels=['Fall', 'Winter', 'Spring', 'Summer']
    )
    
    return df

# Create a wide format dataset with one row per patient-test date
def create_patient_test_data(df):
    # Group by patient ID and test date
    df_wide = df.pivot_table(
        index=['StudyID', 'YEAR', 'lab_days'],
        columns='pathogen_group',
        values='is_positive',
        aggfunc='max',  # If multiple tests on same day, consider positive if any is positive
        fill_value=0
    ).reset_index()
    
    # Add season and time period variables
    temp_df = df[['StudyID', 'YEAR', 'lab_days', 'season', 'week_within_year', 'month_within_year', 
                  'year_week', 'year_month']].drop_duplicates()
    df_wide = df_wide.merge(temp_df, on=['StudyID', 'YEAR', 'lab_days'], how='left')
    
    return df_wide

# Function to perform chi-square test for each virus pair
def chi_square_virus_pairs(wide_df, pathogens_of_interest):
    results = []
    
    for i, virus1 in enumerate(pathogens_of_interest):
        for virus2 in pathogens_of_interest[i+1:]:
            # Create contingency table
            contingency = pd.crosstab(wide_df[virus1], wide_df[virus2])
            
            # Perform chi-square test
            chi2, p, dof, expected = chi2_contingency(contingency)
            
            # Calculate observed vs expected for co-infection cell
            observed = contingency.iloc[1, 1]
            expected_value = expected[1, 1]
            ratio = observed / expected_value if expected_value > 0 else float('inf')
            
            results.append({
                'Virus1': virus1,
                'Virus2': virus2,
                'Chi-square': chi2,
                'p-value': p,
                'Observed Co-infections': observed,
                'Expected Co-infections': expected_value,
                'O/E Ratio': ratio
            })
    
    return pd.DataFrame(results)

# Stratified analysis by time period
def stratified_chi_square(wide_df, pathogens_of_interest, time_var='month_within_year', year_specific=True):
    """
    Perform chi-square tests stratified by time period
    
    Parameters:
    -----------
    wide_df : DataFrame
        Wide-format dataframe with one row per patient-test
    pathogens_of_interest : list
        List of pathogen names to analyze
    time_var : str
        Variable to use for time stratification ('week_within_year', 'month_within_year', 'season')
    year_specific : bool
        If True, use year-specific time periods (e.g., year_week or year_month)
        If False, pool the same time periods across years
    """
    # Choose the appropriate time variable
    if year_specific:
        if time_var == 'week_within_year':
            strata_var = 'year_week'
        elif time_var == 'month_within_year':
            strata_var = 'year_month'
        else:
            strata_var = 'YEAR' + '_' + time_var
    else:
        strata_var = time_var
    
    time_periods = wide_df[strata_var].unique()
    results = []
    
    for period in time_periods:
        period_df = wide_df[wide_df[strata_var] == period]
        
        # Skip time periods with too few samples
        if len(period_df) < 20:  # Minimum sample size threshold
            continue
            
        for i, virus1 in enumerate(pathogens_of_interest):
            for virus2 in pathogens_of_interest[i+1:]:
                # Skip if there are too few positive cases in this time period
                if period_df[virus1].sum() < 5 or period_df[virus2].sum() < 5:
                    continue
                    
                # Create contingency table
                contingency = pd.crosstab(period_df[virus1], period_df[virus2])
                
                # Perform chi-square test
                chi2, p, dof, expected = chi2_contingency(contingency)
                
                # Calculate observed vs expected for co-infection cell
                observed = contingency.iloc[1, 1] if 1 in contingency.index and 1 in contingency.columns else 0
                expected_value = expected[1, 1] if expected.shape == (2, 2) else 0
                ratio = observed / expected_value if expected_value > 0 else float('inf')
                
                results.append({
                    'Time_Period': period,
                    'Virus1': virus1,
                    'Virus2': virus2,
                    'Chi-square': chi2,
                    'p-value': p,
                    'Observed Co-infections': observed,
                    'Expected Co-infections': expected_value,
                    'O/E Ratio': ratio,
                    'Sample_Size': len(period_df)
                })
    
    return pd.DataFrame(results)

# Mantel-Haenszel test to combine across time periods
def mantel_haenszel_test(wide_df, pathogens_of_interest, time_var='month_within_year', year_specific=True):
    """
    Perform Mantel-Haenszel test to combine evidence across time periods
    
    Parameters:
    -----------
    wide_df : DataFrame
        Wide-format dataframe with one row per patient-test
    pathogens_of_interest : list
        List of pathogen names to analyze
    time_var : str
        Variable to use for time stratification ('week_within_year', 'month_within_year', 'season')
    year_specific : bool
        If True, use year-specific time periods (e.g., year_week or year_month)
        If False, pool the same time periods across years
    """
    # Choose the appropriate time variable
    if year_specific:
        if time_var == 'week_within_year':
            strata_var = 'year_week'
        elif time_var == 'month_within_year':
            strata_var = 'year_month'
        else:
            strata_var = 'YEAR' + '_' + time_var
    else:
        strata_var = time_var
    
    time_periods = wide_df[strata_var].unique()
    results = []
    
    for i, virus1 in enumerate(pathogens_of_interest):
        for virus2 in pathogens_of_interest[i+1:]:
            total_a = 0  # virus1+, virus2+
            total_b = 0  # virus1+, virus2-
            total_c = 0  # virus1-, virus2+
            total_d = 0  # virus1-, virus2-
            
            # Weight for each stratum (time period)
            weights = []
            odds_ratios = []
            
            for period in time_periods:
                period_df = wide_df[wide_df[strata_var] == period]
                
                # Skip periods with too few samples
                if len(period_df) < 20:
                    continue
                
                # Create contingency table
                table = pd.crosstab(period_df[virus1], period_df[virus2])
                
                # Extract values (with safety checks)
                a = table.loc[1, 1] if 1 in table.index and 1 in table.columns else 0
                b = table.loc[1, 0] if 1 in table.index and 0 in table.columns else 0
                c = table.loc[0, 1] if 0 in table.index and 1 in table.columns else 0
                d = table.loc[0, 0] if 0 in table.index and 0 in table.columns else 0
                
                # Add to totals
                total_a += a
                total_b += b
                total_c += c
                total_d += d
                
                # Calculate odds ratio for this stratum
                # Add 0.5 to each cell if any cell is 0 (Haldane correction)
                if a*b*c*d == 0:
                    a += 0.5
                    b += 0.5
                    c += 0.5
                    d += 0.5
                
                or_stratum = (a * d) / (b * c) if b * c > 0 else float('inf')
                odds_ratios.append(or_stratum)
                
                # Calculate weight
                weight = 1 / ((1/a) + (1/b) + (1/c) + (1/d)) if a*b*c*d > 0 else 0
                weights.append(weight)
            
            # Calculate weighted average odds ratio
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_or = sum(w * or_s for w, or_s in zip(weights, odds_ratios)) / total_weight
            else:
                weighted_or = (total_a * total_d) / (total_b * total_c) if total_b * total_c > 0 else float('inf')
            
            # Simple calculation of overall odds ratio
            simple_or = (total_a * total_d) / (total_b * total_c) if total_b * total_c > 0 else float('inf')
            
            results.append({
                'Virus1': virus1,
                'Virus2': virus2,
                'Time_Variable': strata_var,
                'Weighted Odds Ratio': weighted_or,
                'Simple Odds Ratio': simple_or,
                'Total Co-infections': total_a,
                'Virus1 Only': total_b,
                'Virus2 Only': total_c,
                'Neither Virus': total_d
            })
    
    return pd.DataFrame(results)

# Logistic regression approach
def logistic_regression_analysis(wide_df, pathogens_of_interest, time_var='month_within_year', year_specific=True):
    """
    Perform logistic regression analysis controlling for time effects
    
    Parameters:
    -----------
    wide_df : DataFrame
        Wide-format dataframe with one row per patient-test
    pathogens_of_interest : list
        List of pathogen names to analyze
    time_var : str
        Variable to use for time control ('week_within_year', 'month_within_year', 'season')
    year_specific : bool
        If True, include year as a separate factor
        If False, pool data across years for the same time periods
    """
    results = []
    
    # Choose the appropriate time variable
    if year_specific and time_var != 'YEAR':
        # Include both time period and year in the model
        model_time_var = time_var
        include_year = True
    else:
        model_time_var = time_var
        include_year = False
    
    for i, virus1 in enumerate(pathogens_of_interest):
        for virus2 in pathogens_of_interest[i+1:]:
            # Prepare data for regression
            model_df = wide_df[[virus1, virus2, model_time_var]].copy()
            
            if include_year:
                model_df['YEAR'] = wide_df['YEAR']
            
            # Create dummy variables for time period
            time_dummies = pd.get_dummies(model_df[model_time_var], prefix=model_time_var, drop_first=True)
            model_df = pd.concat([model_df, time_dummies], axis=1)
            
            # Create year dummies if needed
            if include_year:
                year_dummies = pd.get_dummies(model_df['YEAR'], prefix='year', drop_first=True)
                model_df = pd.concat([model_df, year_dummies], axis=1)
            
            try:
                # Construct formula
                dummy_terms = " + ".join(time_dummies.columns)
                if include_year:
                    dummy_terms += " + " + " + ".join(year_dummies.columns)
                    
                formula = f"{virus2} ~ {virus1} + {dummy_terms}"
                
                # Fit logistic regression model
                model = logit(formula, data=model_df).fit(disp=False)
                
                # Extract coefficient for virus1
                coef = model.params[virus1]
                p_value = model.pvalues[virus1]
                odds_ratio = np.exp(coef)
                ci_lower = np.exp(coef - 1.96 * model.bse[virus1])
                ci_upper = np.exp(coef + 1.96 * model.bse[virus1])
                
                results.append({
                    'Virus1': virus1,
                    'Virus2': virus2,
                    'Time_Variable': model_time_var,
                    'Year_Specific': include_year,
                    'Coefficient': coef,
                    'p-value': p_value,
                    'Odds Ratio': odds_ratio,
                    'CI Lower': ci_lower,
                    'CI Upper': ci_upper
                })
            except Exception as e:
                # Handle convergence issues or other errors
                results.append({
                    'Virus1': virus1,
                    'Virus2': virus2,
                    'Time_Variable': model_time_var,
                    'Year_Specific': include_year,
                    'Coefficient': None,
                    'p-value': None,
                    'Odds Ratio': None,
                    'CI Lower': None,
                    'CI Upper': None,
                    'Error': str(e)
                })
    
    return pd.DataFrame(results)

# Visualize results
def plot_odds_ratios(results_df, title):
    plt.figure(figsize=(10, 6))
    
    # Format for plotting
    plot_data = results_df.copy()
    plot_data['Pair'] = plot_data['Virus1'] + ' + ' + plot_data['Virus2']
    plot_data['log_OR'] = np.log(plot_data['Odds Ratio'])
    
    # Sort by odds ratio
    plot_data = plot_data.sort_values('Odds Ratio')
    
    # Plot
    sns.barplot(x='Pair', y='Odds Ratio', data=plot_data)
    plt.axhline(y=1, color='r', linestyle='--')
    plt.ylabel('Odds Ratio (log scale)')
    plt.xlabel('Virus Pair')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yscale('log')
    plt.tight_layout()
    
    return plt

# Main function to run the analysis
def main(file_path, time_var='month_within_year', year_specific=True):
    """
    Main function to run viral interference analysis
    
    Parameters:
    -----------
    file_path : str
        Path to the SAS dataset
    time_var : str
        Time variable to use for controlling seasonality:
        'week_within_year' - Weeks (more granular)
        'month_within_year' - Months (moderate granularity)
        'season' - Seasons (less granular)
    year_specific : bool
        If True, treat the same time periods in different years as distinct
        If False, pool data from the same time periods across years
    """
    # Load data
    print("Loading data...")
    df = load_sas_data(file_path)
    
    # Process data
    print("Processing data...")
    processed_df = process_data(df)
    
    # Create wide format data
    print("Creating patient-test level data...")
    wide_df = create_patient_test_data(processed_df)
    
    # List of pathogens of interest
    pathogens_of_interest = ['Influenza', 'RSV', 'HMPV', 'Adenovirus']
    
    # Basic summary of the data
    print("\nData Summary:")
    print(f"Total number of patient-test records: {len(wide_df)}")
    for pathogen in pathogens_of_interest:
        pos_count = wide_df[pathogen].sum()
        print(f"{pathogen}: {pos_count} positive tests ({pos_count/len(wide_df)*100:.2f}%)")
    
    # Chi-square test for each virus pair (overall)
    print("\nPerforming overall chi-square tests...")
    chi_square_results = chi_square_virus_pairs(wide_df, pathogens_of_interest)
    print("\nOverall chi-square test results:")
    print(chi_square_results)
    
    # Stratified analysis by time period
    print(f"\nPerforming stratified analysis by {time_var}, year_specific={year_specific}...")
    stratified_results = stratified_chi_square(wide_df, pathogens_of_interest, time_var, year_specific)
    print("\nStratified chi-square test results:")
    print(stratified_results)
    # save to csv
    stratified_results.to_csv("Data/Processed/interference_stratified_chi-square_weekly.csv", index=False)
    
    # Mantel-Haenszel test to combine across time periods
    print("\nPerforming Mantel-Haenszel test to combine across time periods...")
    mh_results = mantel_haenszel_test(wide_df, pathogens_of_interest, time_var, year_specific)
    print("\nMantel-Haenszel test results:")
    print(mh_results)
    
    # Logistic regression analysis
    print("\nPerforming logistic regression analysis...")
    logistic_results = logistic_regression_analysis(wide_df, pathogens_of_interest, time_var, year_specific)
    print("\nLogistic regression results:")
    print(logistic_results)
    
    # Plot results
    print("\nGenerating visualizations...")
    plot = plot_odds_ratios(logistic_results, f"Viral Interference: Adjusted Odds Ratios (Controlling for {time_var})")
    plot.show()
    
    return {
        'chi_square': chi_square_results,
        'stratified': stratified_results,
        'mantel_haenszel': mh_results,
        'logistic': logistic_results,
        'wide_data': wide_df,
        'processed_data': processed_df
    }

# Example usage:
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "Data/Raw/KPSC/testing.sas7bdat"
    
    # Choose time variable for seasonality control:
    # 'week_within_year' - Most granular (weekly)
    # 'month_within_year' - Moderate granularity (monthly)
    # 'season' - Least granular (seasonal)
    time_var = 'week_within_year'
    
    # Whether to treat the same time periods in different years as distinct
    year_specific = True
    
    results = main(file_path, time_var, year_specific)
    
    # Alternatively, run both year-specific and pooled analyses for comparison
    # results_year_specific = main(file_path, 'month_within_year', True)
    # results_pooled = main(file_path, 'month_within_year', False)
