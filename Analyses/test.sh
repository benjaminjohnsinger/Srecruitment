combinations=(
"RSV 250325 FlexStepwise X flexage"
"RSV 250325 FlexStepwise setimport flexage"
"InfluenzaA 250325 FlexStepwise X flexage"
"InfluenzaA 250325 FlexStepwise setimport flexage"
"InfluenzaB 250325 FlexStepwise X flexage"
"InfluenzaB 250325 FlexStepwise setimport flexage"
"Metapneumovirus 250325 FlexStepwise X flexage"
"Metapneumovirus 250325 FlexStepwise setimport flexage"
"Adenovirus 250325 FlexStepwise X flexage"
"Adenovirus 250325 FlexStepwise setimport flexage"
"Parainfluenza3 250325 FlexStepwise X flexage"
"Parainfluenza3 250325 FlexStepwise setimport flexage"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done