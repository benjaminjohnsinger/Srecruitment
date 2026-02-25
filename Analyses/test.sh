combinations=(
"Adenovirus 260224 FlexStepwise NA flexagep01 1e-9 1000 0.003 3"
"Metapneumovirus 260224 FlexStepwise NA flexagep01 1e-9 1000 0.003 3"
"Parainfluenza3 260224 FlexStepwise NA flexagep01 1e-9 1000 0.003 3"
"RSV 260224 FlexStepwise NA flexagep01 1e-9 1000 0.003 3"
"InfluenzaA 260224 FlexStepwise NA flexagep01 1e-9 1000 0.003 3"
"InfluenzaB 260224 FlexStepwise NA flexagep01 1e-9 1000 0.003 3"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
done