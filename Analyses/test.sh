combinations=(
"RSV 250616 FlexStepwise setimport flexage 0.01 20 1 0.7"
"RSV 2506162 FlexStepwise setimport flexage 0.01 20 1 0.7"
# # "RSV 2506163 FlexStepwise setimport flexage 0.01 20 1 0.7"
# # "RSV 2506164 FlexStepwise setimport flexage 0.01 20 1 0.7"
# "InfluenzaA 250421 FlexStepwise setimport flexage"
"InfluenzaA 250616 FlexStepwise setimport flexage 0.01 20 1 0.7"
"InfluenzaA 2506162 FlexStepwise setimport flexage 0.01 20 1 0.7"
# # "InfluenzaA 2506163 FlexStepwise setimport flexage 0.01 20 1 0.7"
# # "InfluenzaA 2506164 FlexStepwise setimport flexage 0.01 20 1 0.7"
# "InfluenzaB 250616 FlexStepwise setimport flexage 0.01 20 1 0.7"
"InfluenzaB 2506162 FlexStepwise setimport flexage 0.01 20 1 0.7"
# # "InfluenzaB 2506163 FlexStepwise setimport flexage 0.01 20 1 0.7"
# # "InfluenzaB 2506164 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Metapneumovirus 250616 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Metapneumovirus 2506162 FlexStepwise setimport flexage 0.01 20 1 0.7"
# "Metapneumovirus 2506163 FlexStepwise setimport flexage 0.01 20 1 0.7"
# # "Metapneumovirus 2506164 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Adenovirus 250616 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Adenovirus 2506162 FlexStepwise setimport flexage 0.01 20 1 0.7"
# "Adenovirus 2506163 FlexStepwise setimport flexage 0.01 20 1 0.7"
# # "Adenovirus 2506164 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Parainfluenza3 250616 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Parainfluenza3 2506162 FlexStepwise setimport flexage 0.01 20 1 0.7"
# "Parainfluenza3 2506163 FlexStepwise setimport flexage 0.01 20 1 0.7"
# # "Parainfluenza3 2506164 FlexStepwise setimport flexage 0.01 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done