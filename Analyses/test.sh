combinations=(
"RSV 260317 Exponential mimm flexagep01 1e-9 20 1000 0.7"
"RSV 260317 FlexStepwise mimm flexagep01 1e-9 20 1000 0.7"
"InfluenzaA 260317 Exponential mimm flexagep01 1e-9 20 1000 0.7"
"Metapneumovirus 260317 Exponential mimm flexagep01 1e-9 20 1000 0.7"
"Parainfluenza3 260317 Exponential mimm flexagep01 1e-9 20 1000 0.7"
"Adenovirus 260317 Exponential mimm flexagep01 1e-9 20 1000 0.7"
"InfluenzaB 260317 Exponential mimm flexagep01 1e-9 20 1000 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
done
for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done