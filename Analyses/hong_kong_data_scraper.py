import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# The main CHP index page
base_url = "https://www.chp.gov.hk/en/statistics/data/10/641/642/2274.html"
response = requests.get(base_url)
soup = BeautifulSoup(response.text, 'html.parser')

# Target years
target_years = ['2015','2016','2017','2018','2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026']

all_data = []

# # Find all hyperlinks on the page
# for link in soup.find_all('a', href=True):
#     link_text = link.text.strip()
    
#     # Check if the link matches our target years
#     if link_text in target_years:
#         full_url = urljoin(base_url, link['href'])
#         print(f"Scraping {link_text} data from: {full_url}")
        
#         try:
#             # pd.read_html automatically finds tables on a webpage
#             tables = pd.read_html(full_url)
            
#             if tables:
#                 # The first table on the page contains the weekly data
#                 df = tables[0]
                
#                 # Add metadata columns to keep track of the source
#                 df['Year'] = link_text
#                 df['Source_URL'] = full_url
                
#                 all_data.append(df)
#         except Exception as e:
#             print(f"Could not read table from {full_url}: {e}")

# # Combine all the scraped tables into one master DataFrame
# if all_data:
#     master_df = pd.concat(all_data, ignore_index=True)
    
#     # Save the consolidated data to a CSV
#     master_df.to_csv("Data/Raw/HK_CHP_Respiratory_Data.csv", index=False)
#     print("Data extraction complete. Saved to Data/Raw/HK_CHP_Respiratory_Data.csv")
# else:
#     print("No data extracted.")

# 1. Load the data and skip the sub-header row (row index 1 in the original CSV)
df = pd.read_csv("Data/Raw/HK_CHP_Respiratory_Data.csv", skiprows=[1])

# 2. Define the core columns we want in every file to track time and sample size
core_cols = ['Year', 'Week number', 'Date', 'No. of specimen tested']

# 3. Influenza A changes schema over time on CHP pages:
# 2023+ uses Type A / Type A.1, while earlier years use subtype columns.
def _get_numeric_col(df, col_name):
    if col_name not in df.columns:
        return pd.Series(index=df.index, dtype='float64')
    return pd.to_numeric(df[col_name], errors='coerce')


def _combine_first_numeric(df, col_names):
    combined = pd.Series(index=df.index, dtype='float64')
    for col_name in col_names:
        combined = combined.combine_first(_get_numeric_col(df, col_name))
    return combined


def _parse_chp_week_monday(date_value, year_value):
    if pd.isna(date_value):
        return pd.NaT

    date_text = str(date_value).strip()
    if not date_text:
        return pd.NaT

    # CHP often stores date as week ranges such as "28/12/2025 - 03/01".
    first_part = re.split(r'\s*[-–—]\s*', date_text)[0].strip()

    parsed = pd.to_datetime(first_part, format='%d/%m/%Y', errors='coerce')
    if pd.isna(parsed):
        parsed = pd.to_datetime(first_part, format='%d/%m', errors='coerce')
        if pd.notna(parsed):
            try:
                year_int = int(float(year_value))
                parsed = parsed.replace(year=year_int)
            except (TypeError, ValueError):
                return pd.NaT

    if pd.isna(parsed):
        return pd.NaT

    # Convert week start label to Monday of that epidemiological week.
    return parsed - pd.to_timedelta(parsed.weekday(), unit='D')


def _set_monday_date_index(processed_df):
    indexed_df = processed_df.copy()
    indexed_df = indexed_df.rename(columns={'Date': 'Week_Range'})
    indexed_df['Date'] = indexed_df.apply(
        lambda row: _parse_chp_week_monday(row['Week_Range'], row['Year']),
        axis=1
    )
    indexed_df = indexed_df.dropna(subset=['Date']).sort_values('Date')
    return indexed_df.set_index('Date')


influenza_a_count_direct = _get_numeric_col(df, 'Type A')
influenza_a_percent_direct = _get_numeric_col(df, 'Type A.1')

influenza_a_subtype_count_cols = ['Type A subtype H1', 'Type A subtype H3', 'Type A subtype H5', 'Type A subtype H7', 'Type A subtype H9']
influenza_a_subtype_percent_cols = ['Type A subtype H1.1', 'Type A subtype H3.1', 'Type A subtype H5.1', 'Type A subtype H7.1', 'Type A subtype H9.1']

influenza_a_count_from_subtypes = pd.concat(
    [_get_numeric_col(df, col) for col in influenza_a_subtype_count_cols],
    axis=1
).sum(axis=1, min_count=1)

influenza_a_percent_from_subtypes = pd.concat(
    [_get_numeric_col(df, col) for col in influenza_a_subtype_percent_cols],
    axis=1
).sum(axis=1, min_count=1)

influenza_a_df = df[core_cols].copy()
influenza_a_df['Influenza_A_Positive_Count'] = influenza_a_count_direct.combine_first(influenza_a_count_from_subtypes)
influenza_a_df['Influenza_A_Percent_Positivity'] = influenza_a_percent_direct.combine_first(influenza_a_percent_from_subtypes)
influenza_a_df = influenza_a_df.dropna(subset=['Influenza_A_Positive_Count', 'Influenza_A_Percent_Positivity'], how='all')
influenza_a_df = _set_monday_date_index(influenza_a_df)
influenza_a_df.to_csv('Data/Processed/HK_CHP_Influenza_A.csv', index_label='Date')
print('Saved Data/Processed/HK_CHP_Influenza_A.csv')

# 4. Influenza B schema also changes over time:
# 2025+ uses Type B / Type B.1, while earlier years store counts in Type B.3.
influenza_b_count = _combine_first_numeric(df, ['Type B', 'Type B.3'])
influenza_b_percent = _combine_first_numeric(df, ['Type B.1'])

influenza_b_df = df[core_cols].copy()
influenza_b_df['Influenza_B_Positive_Count'] = influenza_b_count
influenza_b_df['Influenza_B_Percent_Positivity'] = influenza_b_percent
influenza_b_df = influenza_b_df.dropna(subset=['Influenza_B_Positive_Count', 'Influenza_B_Percent_Positivity'], how='all')
influenza_b_df = _set_monday_date_index(influenza_b_df)
influenza_b_df.to_csv('Data/Processed/HK_CHP_Influenza_B.csv', index_label='Date')
print('Saved Data/Processed/HK_CHP_Influenza_B.csv')

# 5. Define a dictionary mapping the remaining standard pathogens to their
# count and percentage column candidates. The first available candidate is used.
pathogen_map = {
    'RSV': {
        'Count_Candidates': ['RSV'],
        'Percent_Candidates': ['RSV.1']
    },
    'Adenovirus': {
        'Count_Candidates': ['Adenovirus'],
        'Percent_Candidates': ['Adenovirus.1']
    },
    'Metapneumovirus': {
        'Count_Candidates': ['Human metapneumovirus', 'Metapneumovirus'],
        'Percent_Candidates': ['Human metapneumovirus.1', 'Metapneumovirus.1']
    }
}

# 6. Extract, rename, and save the standard pathogens
for pathogen, cols in pathogen_map.items():
    pathogen_count = _combine_first_numeric(df, cols['Count_Candidates'])
    pathogen_percent = _combine_first_numeric(df, cols['Percent_Candidates'])

    pathogen_df = df[core_cols].copy()
    pathogen_df[f"{pathogen}_Positive_Count"] = pathogen_count
    pathogen_df[f"{pathogen}_Percent_Positivity"] = pathogen_percent
    pathogen_df = pathogen_df.dropna(
        subset=[f"{pathogen}_Positive_Count", f"{pathogen}_Percent_Positivity"],
        how='all'
    )
    pathogen_df = _set_monday_date_index(pathogen_df)
    
    # Save to a separate CSV
    pathogen_df.to_csv(f"Data/Processed/HK_CHP_{pathogen}.csv", index_label='Date')
    print(f"Saved Data/Processed/HK_CHP_{pathogen}.csv")

# 7. Handle Parainfluenza separately since it is split into 4 subtypes on the CHP site
parainfluenza_cols = {
    'Type_1_Count': 'Parainfluenza 1', 'Type_1_Percent': 'Parainfluenza 1.1',
    'Type_2_Count': 'Parainfluenza 2', 'Type_2_Percent': 'Parainfluenza 2.1',
    'Type_3_Count': 'Parainfluenza 3', 'Type_3_Percent': 'Parainfluenza 3.1',
    'Type_4_Count': 'Parainfluenza 4', 'Type_4_Percent': 'Parainfluenza 4.1'
}

para_extract_cols = core_cols + list(parainfluenza_cols.values())
para_df = df[para_extract_cols].dropna(subset=list(parainfluenza_cols.values()), how='all').copy()

# Rename parainfluenza columns for clarity
para_df.rename(columns={v: k for k, v in parainfluenza_cols.items()}, inplace=True)
para_df = _set_monday_date_index(para_df)

# Save the parainfluenza dataset
para_df.to_csv("Data/Processed/HK_CHP_Parainfluenza.csv", index_label='Date')
print("Saved Data/Processed/HK_CHP_Parainfluenza.csv")

# six panel plot of positive counts for Influenza A, Influenza B, RSV, Adenovirus,
# Metapneumovirus, and Parainfluenza 3 over time (x-axis is date) with one panel
# for each pathogen. Save the figure as
# "Figures/HK_CHP_Pathogen_Positive_Counts_Over_Time.png" with a dpi of 300.
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as pe


plot_specs = [
    {
        'name': 'Influenza A',
        'csv': 'Data/Processed/HK_CHP_Influenza_A.csv',
        'y_col': 'Influenza_A_Positive_Count',
        'color': '#1f77b4'
    },
    {
        'name': 'Influenza B',
        'csv': 'Data/Processed/HK_CHP_Influenza_B.csv',
        'y_col': 'Influenza_B_Positive_Count',
        'color': '#ff7f0e'
    },
    {
        'name': 'RSV',
        'csv': 'Data/Processed/HK_CHP_RSV.csv',
        'y_col': 'RSV_Positive_Count',
        'color': '#2ca02c'
    },
    {
        'name': 'Adenovirus',
        'csv': 'Data/Processed/HK_CHP_Adenovirus.csv',
        'y_col': 'Adenovirus_Positive_Count',
        'color': '#d62728'
    },
    {
        'name': 'Metapneumovirus',
        'csv': 'Data/Processed/HK_CHP_Metapneumovirus.csv',
        'y_col': 'Metapneumovirus_Positive_Count',
        'color': '#9467bd'
    },
    {
        'name': 'Parainfluenza 3',
        'csv': 'Data/Processed/HK_CHP_Parainfluenza.csv',
        'y_col': 'Type_3_Count',
        'color': '#8c564b'
    }
]


def _annotate_suppression_duration(ax, plot_df, y_col):
    monthly = (
        plot_df[['Date', y_col]]
        .dropna()
        .set_index('Date')
        .resample('MS')
        .sum()
    )

    if monthly.empty:
        return

    threshold = monthly[y_col].max() / 25
    dip_time = monthly[
        (monthly.index > pd.to_datetime('2020-01-01'))
        & (monthly[y_col] < threshold)
    ].index.min()

    if pd.isna(dip_time):
        return

    dip_idx = monthly.index.get_loc(dip_time)
    last_pre_time = monthly.index[max(dip_idx - 1, 0)]
    rebound_time = monthly[
        (monthly.index > last_pre_time)
        & (monthly[y_col] > threshold)
    ].index.min()

    if pd.isna(rebound_time):
        end_time = monthly.index.max()
        y_level = monthly.loc[last_pre_time, y_col]
        time_diff = end_time - last_pre_time
        month_count = max(time_diff.days // 30, 1)
        label = f"{month_count}+ months"
        ax.plot([last_pre_time, end_time], [y_level, y_level], color='black', linewidth=1.1)
        ax.annotate(
            label,
            xy=(last_pre_time + time_diff / 2, y_level),
            xytext=(0, 2),
            textcoords='offset points',
            ha='center',
            va='bottom',
            color='black',
            path_effects=[pe.Stroke(linewidth=1, foreground='white'), pe.Normal()],
        )
        return

    time_diff = rebound_time - last_pre_time
    month_count = max(time_diff.days // 30, 1)
    label = f"{month_count} months"
    y_level = monthly.loc[rebound_time, y_col]

    ax.plot([last_pre_time, rebound_time], [y_level, y_level], color='black', linewidth=1.1)
    ax.annotate(
        label,
        xy=(last_pre_time + time_diff / 2, y_level),
        xytext=(0, 2),
        textcoords='offset points',
        ha='center',
        va='bottom',
        color='black',
        path_effects=[pe.Stroke(linewidth=1, foreground='white'), pe.Normal()],
    )

sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
axes = axes.flatten()

for ax, spec in zip(axes, plot_specs):
    plot_df = pd.read_csv(spec['csv'], parse_dates=['Date']).copy()
    plot_df.loc[:, spec['y_col']] = pd.to_numeric(plot_df[spec['y_col']], errors='coerce')
    plot_df = plot_df.dropna(subset=['Date', spec['y_col']]).sort_values('Date')

    sns.lineplot(
        data=plot_df,
        x='Date',
        y=spec['y_col'],
        ax=ax,
        color=spec['color'],
        linewidth=1.6
    )
    ax.set_title(spec['name'], fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Positive cases detected')
    _annotate_suppression_duration(ax, plot_df, spec['y_col'])

fig.suptitle('Hong Kong CHP Respiratory Pathogen Positive Counts Over Time', fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig('Figures/HK_CHP_Pathogen_Positive_Counts_Over_Time.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Saved Figures/HK_CHP_Pathogen_Positive_Counts_Over_Time.png')