combinations=(
"RSV 251103 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2511032 FlexStepwise NA flexage 1e-9 20 1 0.7"
"InfluenzaA 2511032 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
"InfluenzaA 251103 FlexStepwise NA flexage 1e-9 20 1 0.7"
"InfluenzaB 2511042 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Adenovirus 2511032 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2511032 FlexStepwise NA flexage 1e-9 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done