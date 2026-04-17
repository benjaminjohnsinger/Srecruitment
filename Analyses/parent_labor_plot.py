import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the folder where your data is stored
data_dir = 'Data/Raw/CensusB23008'

# List to keep track of the yearly data
results = []

# Loop through the years you downloaded
for year in range(2010, 2025):
    # Adjust this string if your filenames look slightly different (e.g. 5-year estimates)
    filename = f'ACSDT1Y{year}.B23008-Data.csv'
    file_path = os.path.join(data_dir, filename)
    
    if os.path.exists(file_path):
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Census CSVs usually have a descriptive text row at index 0. 
        # We drop it so we can do math on the actual numbers.
        df_data = df.drop(0)
        
        # Columns needed for the calculation
        cols = ['B23008_002E', 'B23008_004E', 'B23008_010E', 'B23008_013E']
        
        # Convert the columns from text to numbers
        for col in cols:
            df_data[col] = pd.to_numeric(df_data[col], errors='coerce')
            
        # Calculate the totals 
        # (Using .sum() ensures this works whether you have 1 row for the US or 50 rows for states)
        total_under_6 = df_data['B23008_002E'].sum()
        two_parents_both_in_lf = df_data['B23008_004E'].sum()
        father_only_in_lf = df_data['B23008_010E'].sum()
        mother_only_in_lf = df_data['B23008_013E'].sum()
        
        # Calculate the proportion
        all_parents_in_lf = two_parents_both_in_lf + father_only_in_lf + mother_only_in_lf
        
        if total_under_6 > 0:
            percentage = (all_parents_in_lf / total_under_6) * 100
        else:
            percentage = 0
            
        results.append({'Year': year, 'Percentage': percentage})
    else:
        print(f"Warning: Could not find file {filename}")

# Convert results into a DataFrame
df_results = pd.DataFrame(results)

# interpolate 2020
if not df_results[df_results['Year'] == 2020].empty:
    print("2020 data already exists, skipping interpolation.")
else:
    if not df_results[df_results['Year'] == 2019].empty and not df_results[df_results['Year'] == 2021].empty:
        percentage_2019 = df_results[df_results['Year'] == 2019]['Percentage'].values[0]
        percentage_2021 = df_results[df_results['Year'] == 2021]['Percentage'].values[0]
        percentage_2020 = (percentage_2019 + percentage_2021) / 2
        df_results = pd.concat([df_results, pd.DataFrame({'Year': [2020], 'Percentage': [percentage_2020]})], ignore_index=True)
        df_results = df_results.sort_values(by='Year').reset_index(drop=True)
    else:
        print("Cannot interpolate 2020 data due to missing adjacent years.")

# print values as list
print(df_results['Percentage'].tolist())

# --- Plotting the Data ---
plt.figure(figsize=(12, 6))

# Plot all data points
plt.plot(df_results['Year'], df_results['Percentage'], marker='o', linestyle='-', color='k', linewidth=2)

# Highlight 2020 with an empty circle to represent interpolated data
if not df_results[df_results['Year'] == 2020].empty:
    year_2020 = df_results[df_results['Year'] == 2020]
    plt.plot(year_2020['Year'], year_2020['Percentage'], marker='o', markersize=8, 
             markerfacecolor='white', markeredgecolor='k', markeredgewidth=2, linestyle='none')

# Formatting the plot
plt.title('Children Under 6 with All Parents in the Labor Force (California)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(df_results['Year']) # Ensures every year is marked on the X-axis
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.ylim(0, 100)
plt.yticks(range(0, 101, 10))

# panel inside plot to zoom in on relevant y range
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
ax = plt.gca()
ax_inset = inset_axes(ax, width="50%", height="50%", loc='lower right', borderpad=2)
ax_inset.plot(df_results['Year'], df_results['Percentage'], marker='o', linestyle='-', color='k', linewidth=2)
if not df_results[df_results['Year'] == 2020].empty:
    ax_inset.plot(year_2020['Year'], year_2020['Percentage'], marker='o', markersize=8, 
                  markerfacecolor='white', markeredgecolor='k', markeredgewidth=2, linestyle='none')
ax_inset.set_xlim(2014, 2025)
ax_inset.set_ylim(60, 70)
ax_inset.set_xticks(range(2014, 2026, 2))
ax_inset.set_yticks(range(60, 71, 5))
ax_inset.set_title('Detail', fontsize=10)
# ax_inset.grid(axis='y', linestyle='--', alpha=0.7)

# plt.tight_layout()
plt.savefig('Figures/parent_labor_plot.png', dpi=300)