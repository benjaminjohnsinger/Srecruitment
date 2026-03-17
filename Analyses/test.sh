combinations=(
"Metapneumovirus 260313 Exponential NA flexagep01 1e-9 200 1000 0.7"
"Metapneumovirus 2603096 Exponential NA flexagep01 1e-9 200 1000 0.7"
# "Adenovirus 260313 Exponential NA flexagep01 1e-9 200 1000 0.7"
# "Parainfluenza3 260313 Exponential NA flexagep01 1e-9 200 1000 0.7"
# "InfluenzaA 260313 Exponential NA flexagep01 1e-9 200 1000 0.7"
# "InfluenzaB 260313 Exponential NA flexagep01 1e-9 200 1000 0.7"
# "Metapneumovirus 2603132 Exponential NA flexagep01 1e-9 200 1000 0.7"
# "Adenovirus 2603132 Exponential NA flexagep01 1e-9 200 1000 0.7"
# "Parainfluenza3 2603132 Exponential NA flexagep01 1e-9 200 1000 0.7"
# "InfluenzaA 2603132 Exponential NA flexagep01 1e-9 200 1000 0.7"
# "InfluenzaB 2603132 Exponential NA flexagep01 1e-9 200 1000 0.7"
)

# for combination in "${combinations[@]}"; do
#     /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
# done
for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done