combinations=(
"RSV 260324 Sigmoid NA flexagep01 1e-9 20 400 0.7"
# "InfluenzaA 260324 Sigmoid NA flexagep01 1e-9 20 400 0.7"
# "InfluenzaB 260324 Sigmoid NA flexagep01 1e-9 20 400 0.7"
# "Metapneumovirus 260324 Sigmoid NA flexagep01 1e-9 20 400 0.7"
# "Parainfluenza3 260324 Sigmoid NA flexagep01 1e-9 20 400 0.7"
# "Adenovirus 260324 Sigmoid NA flexagep01 1e-9 20 400 0.7"
)

for combination in "${combinations[@]}"; do
    # /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
    if [[ $combination == *"optax" ]]; then
        /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_optax.py $combination
    else
        /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
    fi
done