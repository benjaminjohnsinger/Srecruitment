combinations=(
"InfluenzaA 251031 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
"InfluenzaA 2510312 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
"Metapneumovirus 251031 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
"Metapneumovirus 2510312 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done