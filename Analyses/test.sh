combinations=(
"InfluenzaA 2507092 FlexStepwise NA flexage 1e-9 20 1 0.7"
"InfluenzaB 2507092 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2507092 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Adenovirus 2507092 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2507092 FlexStepwise NA flexage 1e-9 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done