#!/bin/bash
## Command(s) to run:

# load venv
source /Users/bjsinger/Documents/Srecruitment/.venv/bin/activate

combinations=(
"RSV 250918 Taube NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 250918 Taube NA flexage 1e-9 20 1 0.7"
"Adenovirus 250918 Taube NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 250918 Taube NA flexage 1e-9 20 1 0.7"
"InfluenzaA 250918 Taube NA flexage 1e-9 20 1 0.7"
"InfluenzaB 250918 Taube NA flexage 1e-9 20 1 0.7"
"RSV 2509182 Taube NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2509182 Taube NA flexage 1e-9 20 1 0.7"
"Adenovirus 2509182 Taube NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2509182 Taube NA flexage 1e-9 20 1 0.7"
"InfluenzaA 2509182 Taube NA flexage 1e-9 20 1 0.7"
"InfluenzaB 2509182 Taube NA flexage 1e-9 20 1 0.7"
"RSV 250918 Mobility NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 250918 Mobility NA flexage 1e-9 20 1 0.7"
"Adenovirus 250918 Mobility NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 250918 Mobility NA flexage 1e-9 20 1 0.7"
"InfluenzaA 250918 Mobility NA flexage 1e-9 20 1 0.7"
"InfluenzaB 250918 Mobility NA flexage 1e-9 20 1 0.7"
"RSV 2509182 Mobility NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2509182 Mobility NA flexage 1e-9 20 1 0.7"
"Adenovirus 2509182 Mobility NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2509182 Mobility NA flexage 1e-9 20 1 0.7"
"InfluenzaA 2509182 Mobility NA flexage 1e-9 20 1 0.7"
"InfluenzaB 2509182 Mobility NA flexage 1e-9 20 1 0.7"
"RSV 250918 Mobility2 NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 250918 Mobility2 NA flexage 1e-9 20 1 0.7"
"Adenovirus 250918 Mobility2 NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 250918 Mobility2 NA flexage 1e-9 20 1 0.7"
"InfluenzaA 250918 Mobility2 NA flexage 1e-9 20 1 0.7"
"InfluenzaB 250918 Mobility2 NA flexage 1e-9 20 1 0.7"
"RSV 2509182 Mobility2 NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2509182 Mobility2 NA flexage 1e-9 20 1 0.7"
"Adenovirus 2509182 Mobility2 NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2509182 Mobility2 NA flexage 1e-9 20 1 0.7"
"InfluenzaA 2509182 Mobility2 NA flexage 1e-9 20 1 0.7"
"InfluenzaB 2509182 Mobility2 NA flexage 1e-9 20 1 0.7"
)

for combination in "${combinations[@]}"; do
   echo "Running combination: $combination"
   # run fit_opt and print outputs to csv file with name based on combination
   /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination >> DE_mobility_tests250918.csv
done

/Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_MCMC.py >> NumPyro_tests250918.txt