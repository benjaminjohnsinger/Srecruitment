combinations=(
"RSV 250625 FlexStepwise 0.05 flexage 0.01 20 1 0.7"
"InfluenzaA 250625 FlexStepwise 0.05 flexage 0.01 20 1 0.7"
"InfluenzaB 250625 FlexStepwise 0.05 flexage 0.01 20 1 0.7"
"Metapneumovirus 250625 FlexStepwise 0.05 flexage 0.01 20 1 0.7"
"Adenovirus 250625 FlexStepwise 0.05 flexage 0.01 20 1 0.7"
"Parainfluenza3 250625 FlexStepwise 0.05 flexage 0.01 20 1 0.7"
"RSV 2506252 FlexStepwise 0.005 flexage 0.01 20 1 0.7"
"InfluenzaA 2506252 FlexStepwise 0.03 flexage 0.01 20 1 0.7"
"InfluenzaB 2506252 FlexStepwise 0.005 flexage 0.01 20 1 0.7"
"Metapneumovirus 2506252 FlexStepwise 0.005 flexage 0.01 20 1 0.7"
"Adenovirus 2506252 FlexStepwise 0.005 flexage 0.01 20 1 0.7"
"Parainfluenza3 2506252 FlexStepwise 0.005 flexage 0.01 20 1 0.7"
"RSV 2506253 FlexStepwise 0.003 flexage 0.01 20 1 0.7"
"InfluenzaA 2506253 FlexStepwise 0.01 flexage 0.01 20 1 0.7"
"InfluenzaB 2506253 FlexStepwise 0.003 flexage 0.01 20 1 0.7"
"Metapneumovirus 2506253 FlexStepwise 0.003 flexage 0.01 20 1 0.7"
"Adenovirus 2506253 FlexStepwise 0.003 flexage 0.01 20 1 0.7"
"Parainfluenza3 2506253 FlexStepwise 0.003 flexage 0.01 20 1 0.7"
"RSV 2506254 FlexStepwise 0.001 flexage 0.01 20 1 0.7"
"InfluenzaA 2506254 FlexStepwise 0.005 flexage 0.01 20 1 0.7"
"InfluenzaB 2506254 FlexStepwise 0.001 flexage 0.01 20 1 0.7"
"Metapneumovirus 2506254 FlexStepwise 0.001 flexage 0.01 20 1 0.7"
"Adenovirus 2506254 FlexStepwise 0.001 flexage 0.01 20 1 0.7"
"Parainfluenza3 2506254 FlexStepwise 0.001 flexage 0.01 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done