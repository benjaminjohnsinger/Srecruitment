combinations=(
# "RSV 250429 250501 setimport maternal 0.01 15 1 0.7"
# "InfluenzaA 250429 250501 setimport maternal 0.01 15 1 0.7"
# "InfluenzaB 250407 250501 setimport flexage 0.01 15 1 0.7"
# "Metapneumovirus 250407 250501 setimport flexage 0.01 15 1 0.7"
"Adenovirus 250513 FlexStepwise setimport flexage 0.01 15 1 0.7"
"Parainfluenza3 250513 FlexStepwise setimport flexage 0.01 15 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done