combinations=(
"test 251015 FlexStepwise NA flexage 1e-9 20 1 0.7"
"test 2510152 FlexStepwise NA flexage 1e-9 20 1 0.7"
"test 2510153 FlexStepwise NA flexage 1e-9 20 1 0.7"
"test 2510154 FlexStepwise NA flexage 1e-9 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done