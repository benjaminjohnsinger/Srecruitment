combinations=(
"RSV 260313 Exponential maternal flexagep01 1e-9 20 1000 0.7"
"RSV 2603132 Exponential maternal flexagep01 1e-9 20 1000 0.7"
"RSV 260313 FlexStepwise maternal flexagep01 1e-9 20 1000 0.7"
"RSV 2603132 FlexStepwise maternal flexagep01 1e-9 20 1000 0.7"
"InfluenzaA 260313 Exponential maternal flexagep01 1e-9 20 1000 0.7"
"InfluenzaA 2603132 Exponential maternal flexagep01 1e-9 20 1000 0.7"
"Metapneumovirus 260313 Exponential maternal flexagep01 1e-9 20 1000 0.7"
"Metapneumovirus 2603132 Exponential maternal flexagep01 1e-9 20 1000 0.7"
"Parainfluenza3 260313 Exponential maternal flexagep01 1e-9 20 1000 0.7"
"Parainfluenza3 2603132 Exponential maternal flexagep01 1e-9 20 1000 0.7"
"Adenovirus 260313 Exponential maternal flexagep01 1e-9 20 1000 0.7"
"Adenovirus 2603132 Exponential maternal flexagep01 1e-9 20 1000 0.7"
"InfluenzaB 260313 Exponential maternal flexagep01 1e-9 20 1000 0.7"
"InfluenzaB 2603132 Exponential maternal flexagep01 1e-9 20 1000 0.7"
"Metapneumovirus 2603134 Exponential NA flexagep01 1e-9 20 1000 0.1"
"Metapneumovirus 2603135 Exponential NA flexagep01 1e-9 20 1000 0.1"
"Metapneumovirus 2603136 Exponential NA flexagep01 1e-9 20 1000 0.9"
"Metapneumovirus 2603137 Exponential NA flexagep01 1e-9 20 1000 0.9"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
done
for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done