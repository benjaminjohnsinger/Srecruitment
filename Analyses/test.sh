combinations=(
"4test0 2510242 FlexStepwise NA flexage 1e-9 20 1 0.7"
"5test0 2510242 FlexStepwise NA flexage 1e-9 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done