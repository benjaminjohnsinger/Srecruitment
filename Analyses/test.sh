combinations=(
"RSV 251105 FlexStepwise wane flexage 1e-9 20 1 0.7"
"RSV 2511052 FlexStepwise wane flexage 1e-9 20 1 0.7"
"Metapneumovirus 251105 FlexStepwise wane flexage 1e-9 20 1 0.7"
"Metapneumovirus 2511052 FlexStepwise wane flexage 1e-9 20 1 0.7"
"InfluenzaA 251105 FlexStepwise wane flexage 1e-9 20 1 0.7"
"InfluenzaA 2511052 FlexStepwise wane flexage 1e-9 20 1 0.7"
"Ademovirus 251105 FlexStepwise wane flexage 1e-9 20 1 0.7"
"Ademovirus 2511052 FlexStepwise wane flexage 1e-9 20 1 0.7"
"Parainfluenza3 251105 FlexStepwise wane flexage 1e-9 20 1 0.7"
"Parainfluenza3 2511052 FlexStepwise wane flexage 1e-9 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done