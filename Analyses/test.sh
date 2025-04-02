combinations=(
"RSV 250321 FlexStepwise X X"
"InfluenzaA 250321 FlexStepwise X X"
"InfluenzaB 250321 FlexStepwise X X"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done