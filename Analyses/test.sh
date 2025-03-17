combinations=(
"InfluenzaA 250306 Mobility ni nr 15 1 0.7"
"InfluenzaA 250306 Stepwise ni nr 15 1 0.7"
"InfluenzaA 250306 Mobility ni nr 15 1 0.7"

"RSV 250311 Stepwise nb X 15 1 0.7"
"RSV 250311 Stepwise X X 15 1 0.7"
"RSV 250311 FlexStepwise X X 15 1 0.7"
"RSV 250311 YoungEarly X X 15 1 0.7"
"RSV 250311 YoungEarly nb X 15 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done