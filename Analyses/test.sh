combinations=(
# "RSV 250407 FlexStepwise setimport flexage"
"RSV 250321 FlexStepwise X X"
# "InfluenzaA 250421 FlexStepwise setimport flexage"
# "InfluenzaB 250421 FlexStepwise setimport flexage"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done