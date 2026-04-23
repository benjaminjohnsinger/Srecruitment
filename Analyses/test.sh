combinations=(
"Metapneumovirus 260423 ExponentialODipEqual brm2split maxagep028 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260423 ExponentialODipEqual brm3split maxagep028 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260423 ExponentialODipEqual brm10split maxagep028 1e-9 20 1 0.7 scipy_DE"
"RSV 260423 ExponentialODipEqual brm2split maxagep028 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260423 ExponentialODipEqual brm2split maxagep05 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260423 ExponentialODipEqual brm2split maxagep028 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 260423 ExponentialODipEqual brm2split maxagep05 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 260423 ExponentialODipEqual brm2split maxagep05 1e-9 20 1 0.7 scipy_DE"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
    if [[ $combination == *"optax" ]]; then
        /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_optax.py $combination
    else
        /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
    fi
done

# for i in {1..100}; do
#     echo "Chunk $i"
#     for combination in "${combinations[@]}"; do
#         /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
#         if [[ $combination == *"optax" ]]; then
#             /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_optax.py $combination
#         else
#             /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
#         fi
#     done
# done