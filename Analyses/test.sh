combinations=(
"InfluenzaA 260226 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"InfluenzaB 260226 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"Parainfluenza3 260226 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"RSV 260226 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"Metapneumovirus 260226 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"Adenovirus 260226 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
done