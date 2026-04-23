combinations=(
"Metapneumovirus 260421 ExponentialODipEqual irelsplit maxagep028 1e-9 20 1 0.7 scipy_DE"

"RSV 260421 ExponentialODipEqual irelsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 260421 ExponentialODipEqual irelsplit nrmaxagep05 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 260421 ExponentialODipEqual irelsplit nrmaxagep05 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260421 ExponentialODipEqual irelsplit maxagep05 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260421 ExponentialODipEqual irelsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
)

for combination in "${combinations[@]}"; do
    # /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
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