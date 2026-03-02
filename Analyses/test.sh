combinations=(
"RSV 260301 FlexStepwise smoothedincidence_data flexage 1e-9 20 1 0.7"
"RSV 2603012 FlexStepwise incidence_data flexage 1e-9 20 1 0.7"
"RSV 2603012 FlexStepwise smoothedincidence_data flexage 1e-9 20 1 0.7"
"RSV 2603013 FlexStepwise incidence_data flexage 1e-9 20 1 0.7"
"RSV 2603013 FlexStepwise smoothedincidence_data flexage 1e-9 20 1 0.7"
"InfluenzaA 260301 FlexStepwise incidence_data flexage 1e-9 20 1 0.7"
"InfluenzaB 260301 FlexStepwise incidence_data flexage 1e-9 20 1 0.7"
"Parainfluenza3 260301 FlexStepwise incidence_data flexage 1e-9 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
done