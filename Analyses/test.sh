combinations=(
"RSV 250317 Mobility X X 15 1 0.7"
"RSV 250317 Mobility nb X 15 1 0.7"
"RSV 250317 FlexStepwise X X 15 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done