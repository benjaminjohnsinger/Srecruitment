import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load First Dataset (Hours distribution for those WITH care)
df1 = pd.read_excel("Data/Raw/CHIS/ChildcareHours.xlsx", header=None)
df1_data = df1.iloc[8:].copy()
df1_data.columns = ['Category_Year', 'Percent', 'CI_95', 'Population']
df1_data.dropna(subset=['Category_Year'], inplace=True)
df1_data = df1_data[~df1_data['Category_Year'].str.contains('Total|Created|Source', case=False, na=False)].copy()
df1_data.loc[:,'Year'] = df1_data['Category_Year'].str.extract(r'\((\d{4})\)').astype(float)

def get_midpoint(cat):
    if '10-14' in cat: return 12
    if '15-19' in cat: return 17
    if '20-24' in cat: return 22
    if '25-29' in cat: return 27
    if '30-34' in cat: return 32
    if '35-39' in cat: return 37
    if '40-44' in cat: return 42
    if '45-49' in cat: return 47
    if '50+' in cat: return 55
    return np.nan

df1_data.loc[:,'Midpoint'] = df1_data['Category_Year'].apply(get_midpoint)
df1_data.loc[:,'Percent_Clean'] = pd.to_numeric(df1_data['Percent'].astype(str).str.replace('*', '', regex=False), errors='coerce') / 100.0

hours_results = []
for year, group in df1_data.groupby('Year'):
    group = group.dropna(subset=['Midpoint', 'Percent_Clean'])
    if group.empty: continue
    
    weights = group['Percent_Clean'] / group['Percent_Clean'].sum()
    mean_hours = np.sum(group['Midpoint'] * weights)
    std_dev = np.sqrt(np.sum(weights * (group['Midpoint'] - mean_hours)**2))
    
    # Estimate sample size internally to find SE
    n_eff_list = []
    for _, row in group.iterrows():
        p = row['Percent_Clean']
        ci_str = str(row['CI_95'])
        if '-' in ci_str:
            try:
                low, high = map(float, ci_str.split('-'))
                ci_margin = (high - low) / 200.0 
                if ci_margin > 0 and 0 < p < 1:
                    n_eff = p * (1 - p) / (ci_margin / 1.96)**2
                    n_eff_list.append(n_eff)
            except Exception:
                pass
                
    se_mean = std_dev / np.sqrt(np.median(n_eff_list)) if n_eff_list else 0
    
    hours_results.append({
        'Year': int(year),
        'Mean_Hours_Conditional': mean_hours,
        'SE_Hours_Conditional': se_mean
    })

df_hours = pd.DataFrame(hours_results)

# Define constant for assumed care hours when no regular childcare
ASSUMED_CARE_HOURS = 0

# 2. Load Second Dataset (Proportion of population WITH care)
df2 = pd.read_excel("Data/Raw/CHIS/Childcare.xlsx", header=None)
df2_data = df2.iloc[6:].copy()
df2_data.columns = ['Category_Year', 'Percent', 'CI_95', 'Population']
df2_data.dropna(subset=['Category_Year'], inplace=True)

# Filter for the affirmative demographic
df2_data = df2_data[df2_data['Category_Year'].str.startswith('Have regular child care (', na=False)].copy()
df2_data.loc[:,'Year'] = df2_data['Category_Year'].str.extract(r'\((\d{4})\)').astype(int)
df2_data.loc[:,'p_care'] = pd.to_numeric(df2_data['Percent'], errors='coerce') / 100.0

# Retrieve Standard Error for proportion mathematically
def get_se_p(row):
    ci_str = str(row['CI_95'])
    if '-' in ci_str:
        try:
            low, high = map(float, ci_str.split('-'))
            return (high - low) / 200.0 / 1.96
        except:
            pass
    return 0

df2_data.loc[:,'SE_p_care'] = df2_data.apply(get_se_p, axis=1)
df_prop = df2_data[['Year', 'p_care', 'SE_p_care']]

# 3. Merge Datasets and Compute Adjusted Metrics
df_merged = pd.merge(df_hours, df_prop, on='Year')

# Adjusted Mean: p_care * conditional_hours + (1-p_care) * ASSUMED_CARE_HOURS
df_merged.loc[:,'Adjusted_Mean'] = (df_merged['p_care'] * df_merged['Mean_Hours_Conditional'] + 
                                     (1 - df_merged['p_care']) * ASSUMED_CARE_HOURS)

# Propagate uncertainty (Delta Method) factoring both components
df_merged.loc[:,'Adjusted_SE'] = np.sqrt(
    (df_merged['p_care']**2 * df_merged['SE_Hours_Conditional']**2) + 
    ((df_merged['Mean_Hours_Conditional'] - ASSUMED_CARE_HOURS)**2 * df_merged['SE_p_care']**2)
)

df_merged.loc[:,'Lower_CI'] = (df_merged['Adjusted_Mean'] - 1.96 * df_merged['Adjusted_SE']).clip(lower=0)
df_merged.loc[:,'Upper_CI'] = df_merged['Adjusted_Mean'] + 1.96 * df_merged['Adjusted_SE']

df_merged.to_csv('Adjusted_Average_Weekly_Childcare_Hours.csv', index=False)

# 4. Create Plot
plt.figure(figsize=(10, 6))
plt.errorbar(df_merged['Year'], df_merged['Adjusted_Mean'], 
             yerr=1.96*df_merged['Adjusted_SE'], fmt='-o', capsize=5, 
             color='k', ecolor='grey', linewidth=2,
             label='Overall Avg Weekly Hours (Includes 0s for no care)')

plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Weekly Hours', fontsize=12)
plt.title('Adjusted Average Weekly Hours in Childcare Over Time\n(Overall Population, including zero hours)', fontsize=14)
plt.xticks(df_merged['Year'])
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# Save final graphic
plt.savefig('Figures/childcare_plot.png')