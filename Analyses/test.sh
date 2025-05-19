combinations=(
"RSV 250516 FlexStepwise setimport flexage 0.01 20 1 0.7"
"RSV 2505162 FlexStepwise setimport flexage 0.01 20 1 0.7"
"RSV 2505163 FlexStepwise setimport flexage 0.01 20 1 0.7"
"RSV 2505164 FlexStepwise setimport flexage 0.01 20 1 0.7"
"InfluenzaA 250516 FlexStepwise setimport flexage 0.01 20 1 0.7"
"InfluenzaA 2505162 FlexStepwise setimport flexage 0.01 20 1 0.7"
"InfluenzaA 2505163 FlexStepwise setimport flexage 0.01 20 1 0.7"
"InfluenzaA 2505164 FlexStepwise setimport flexage 0.01 20 1 0.7"
"InfluenzaB 250516 FlexStepwise setimport flexage 0.01 20 1 0.7"
"InfluenzaB 2505162 FlexStepwise setimport flexage 0.01 20 1 0.7"
"InfluenzaB 2505163 FlexStepwise setimport flexage 0.01 20 1 0.7"
"InfluenzaB 2505164 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Metapneumovirus 250516 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Metapneumovirus 2505162 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Metapneumovirus 2505163 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Metapneumovirus 2505164 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Adenovirus 250516 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Adenovirus 2505162 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Adenovirus 2505163 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Adenovirus 2505164 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Parainfluenza3 250516 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Parainfluenza3 2505162 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Parainfluenza3 2505163 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Parainfluenza3 2505164 FlexStepwise setimport flexage 0.01 20 1 0.7"

)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done