combinations=(
"RSV 260317 Exponential NA flexagep01 1e-9 4 4 0.7"
# "Metapneumovirus 260316 Exponential NA flexagep01 1e-9 200 1000 0.7"
# "Adenovirus 260316 Exponential NA flexagep01 1e-9 200 1000 0.7"
# "Parainfluenza3 260316 Exponential NA flexagep01 1e-9 200 1000 0.7"
# "InfluenzaA 260316 Exponential NA flexagep01 1e-9 200 1000 0.7"
# "InfluenzaB 260316 Exponential NA flexagep01 1e-9 200 1000 0.7"

)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
done
for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done