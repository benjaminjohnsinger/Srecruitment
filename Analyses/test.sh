combinations=(
"RSV 250407 FlexStepwise setimport flexage"
"InfluenzaA 250407 FlexStepwise setimport flexage"
"InfluenzaB 250407 FlexStepwise setimport flexage"
"Metapneumovirus 250407 FlexStepwise setimport flexage"
"Adenovirus 250407 FlexStepwise setimport flexage"
"Parainfluenza3 250407 FlexStepwise setimport flexage"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done