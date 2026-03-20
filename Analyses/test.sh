combinations=(
"RSV 260318 Exponential mimm flexagep01 1e-9 20 400 0.7"
"RSV 2603182 Exponential mimm flexagep01 1e-9 20 400 0.7"
"RSV 260318 Exponential maxmimm flexagep01 1e-9 20 400 0.7"
"InfluenzaA 260318 Exponential mimm flexagep01 1e-9 20 400 0.7"
"Metapneumovirus 260318 Exponential mimm flexagep01 1e-9 20 400 0.7"
"Parainfluenza3 260318 Exponential mimm flexagep01 1e-9 20 400 0.7"
"Adenovirus 260318 Exponential mimm flexagep01 1e-9 20 400 0.7"
"InfluenzaB 260318 Exponential mimm flexagep01 1e-9 20 400 0.7"
"RSV 2603183 Exponential mimm flexagep01 1e-9 200 1000 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
done
for combination in "${combinations[@]}"; do
    if [[ $combination == *"optax" ]]; then
        /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_optax.py $combination
    else
        /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
    fi
done