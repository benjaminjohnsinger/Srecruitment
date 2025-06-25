combinations=(
"RSV 250620 FlexStepwise setimport flexage 0.01 20 1 0.7"
"RSV 2506202 FlexStepwise setimport flexage 0.01 20 1 0.7"
"RSV 2506203 FlexStepwise setimport flexage 0.01 20 1 0.7"
"RSV 2506204 FlexStepwise setimport flexage 0.01 20 1 0.7"
"InfluenzaA 250620 FlexStepwise setimport flexage 0.01 20 1 0.7"
"InfluenzaA 2506202 FlexStepwise setimport flexage 0.01 20 1 0.7"
"InfluenzaA 2506203 FlexStepwise setimport flexage 0.01 20 1 0.7"
"InfluenzaA 2506204 FlexStepwise setimport flexage 0.01 20 1 0.7"
"InfluenzaB 250620 FlexStepwise setimport flexage 0.01 20 1 0.7"
"InfluenzaB 2506202 FlexStepwise setimport flexage 0.01 20 1 0.7"
"InfluenzaB 2506203 FlexStepwise setimport flexage 0.01 20 1 0.7"
"InfluenzaB 2506204 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Metapneumovirus 250620 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Metapneumovirus 2506202 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Metapneumovirus 2506203 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Metapneumovirus 2506204 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Adenovirus 250620 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Adenovirus 2506202 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Adenovirus 2506203 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Adenovirus 2506204 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Parainfluenza3 250620 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Parainfluenza3 2506202 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Parainfluenza3 2506203 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Parainfluenza3 2506204 FlexStepwise setimport flexage 0.01 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done