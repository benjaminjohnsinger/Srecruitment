combinations=(
# "InfluenzaA 2508057013 FlexStepwise 0.005 flexage"
# "RSV 250709 FlexStepwise 0.005 flexage"
# "Metapneumovirus 2507092 FlexStepwise 0.005 flexage"
# "Parainfluenza3 250709 FlexStepwise 0.005 flexage"
# "Adenovirus 250709 FlexStepwise 0.005 flexage"
"InfluenzaA 2507092 FlexStepwise 0.005 flexage"
# "InfluenzaB 250709 FlexStepwise 0.005 flexage"
# "RSV 250516 FlexStepwise setimport flexage"
# "Metapneumovirus 250516 FlexStepwise setimport flexage"
# "Parainfluenza3 250516 FlexStepwise setimport flexage"
# "Adenovirus 250516 FlexStepwise setimport flexage"
# "InfluenzaA 250516 FlexStepwise setimport flexage"
# "InfluenzaB 250516 FlexStepwise setimport flexage"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done