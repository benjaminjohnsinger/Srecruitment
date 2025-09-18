#!/bin/bash
## Command(s) to run:

combinations=(
"RSV 250923 Taube NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 250923 Taube NA flexage 1e-9 20 1 0.7"
"Adenovirus 250923 Taube NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 250923 Taube NA flexage 1e-9 20 1 0.7"
"InfluenzaA 250923 Taube NA flexage 1e-9 20 1 0.7"
"InfluenzaB 250923 Taube NA flexage 1e-9 20 1 0.7"
"RSV 2509232 Taube NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2509232 Taube NA flexage 1e-9 20 1 0.7"
"Adenovirus 2509232 Taube NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2509232 Taube NA flexage 1e-9 20 1 0.7"
"InfluenzaA 2509232 Taube NA flexage 1e-9 20 1 0.7"
"InfluenzaB 2509232 Taube NA flexage 1e-9 20 1 0.7"
"RSV 250923 Mobility NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 250923 Mobility NA flexage 1e-9 20 1 0.7"
"Adenovirus 250923 Mobility NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 250923 Mobility NA flexage 1e-9 20 1 0.7"
"InfluenzaA 250923 Mobility NA flexage 1e-9 20 1 0.7"
"InfluenzaB 250923 Mobility NA flexage 1e-9 20 1 0.7"
"RSV 2509232 Mobility NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2509232 Mobility NA flexage 1e-9 20 1 0.7"
"Adenovirus 2509232 Mobility NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2509232 Mobility NA flexage 1e-9 20 1 0.7"
"InfluenzaA 2509232 Mobility NA flexage 1e-9 20 1 0.7"
"InfluenzaB 2509232 Mobility NA flexage 1e-9 20 1 0.7"
"RSV 250923 Mobility2 NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 250923 Mobility2 NA flexage 1e-9 20 1 0.7"
"Adenovirus 250923 Mobility2 NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 250923 Mobility2 NA flexage 1e-9 20 1 0.7"
"InfluenzaA 250923 Mobility2 NA flexage 1e-9 20 1 0.7"
"InfluenzaB 250923 Mobility2 NA flexage 1e-9 20 1 0.7"
"RSV 2509232 Mobility2 NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2509232 Mobility2 NA flexage 1e-9 20 1 0.7"
"Adenovirus 2509232 Mobility2 NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2509232 Mobility2 NA flexage 1e-9 20 1 0.7"
"InfluenzaA 2509232 Mobility2 NA flexage 1e-9 20 1 0.7"
"InfluenzaB 2509232 Mobility2 NA flexage 1e-9 20 1 0.7"
)

for combination in "${combinations[@]}" do
   echo "Running combination: $combination"
   # run fit_opt and print outputs to csv file with name based on combination
   /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/fit_opt.py $combination >> DE_mobility_tests250923.csv
done