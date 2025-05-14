combinations=(
"RSV 250513 FlexStepwise setimport flexage 0.01 15 1 0.7"
# "InfluenzaA 250513 FlexStepwise setimport flexage 0.01 15 1 0.7"
# "RSV 2505132 FlexStepwise setimport flexage 0.01 15 1 0.7"
# "Adenovirus 2505132 FlexStepwise setimport flexage 0.01 15 1 0.7"
# "InfluenzaB 250513 FlexStepwise setimport flexage 0.01 15 1 0.7"
# "Metapneumovirus 250513 FlexStepwise setimport flexage 0.01 15 1 0.7"
# "Adenovirus 250513 FlexStepwise setimport flexage 0.01 15 1 0.7"
# "Parainfluenza3 250513 FlexStepwise setimport flexage 0.01 15 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done