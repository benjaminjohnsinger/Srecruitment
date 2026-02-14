combinations=(
"RSV 260210 FlexStepwise NA flexage 1e-9 20 1 0.7"
"RSV 2602102 FlexStepwise NA flexage 1e-9 20 1 0.7"
"RSV 2602103 FlexStepwise NA flexage 1e-9 20 1 0.7"
"InfluenzaA 260210 FlexStepwise NA flexage 1e-9 20 1 0.7"
"InfluenzaA 2602102 FlexStepwise NA flexage 1e-9 20 1 0.7"
"InfluenzaA 2602103 FlexStepwise NA flexage 1e-9 20 1 0.7"
"InfluenzaB 260210 FlexStepwise NA flexage 1e-9 20 1 0.7"
"InfluenzaB 2602102 FlexStepwise NA flexage 1e-9 20 1 0.7"
"InfluenzaB 2602103 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 260210 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2602102 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2602103 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Adenovirus 260210 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Adenovirus 2602102 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Adenovirus 2602103 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 260210 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2602102 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2602103 FlexStepwise NA flexage 1e-9 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done