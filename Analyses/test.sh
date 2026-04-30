combinations=(
# "RSV 2604283 ExponentialODipEqual dedupsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaA 2604283 ExponentialODipEqual dedupsplit maxagep1 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 2604283 ExponentialODipEqual dedupsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
# "Adenovirus 2604283 ExponentialODipEqual dedupsplit maxagep05 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 2604283 ExponentialODipEqual dedupsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaB 2604283 ExponentialODipEqual dedupsplit maxagep05 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaA 2604283 ExponentialODipEqual dedupsplit nrmaxagep05 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaB 2604283 ExponentialODipEqual dedupsplit nrmaxagep05 1e-9 20 1 0.7 scipy_DE"

# # "Metapneumovirus 260428 ExponentialODipEqual NA cboost2maxagep028 20 1 0.7 1e-9 scipy_DE"
# # "Metapneumovirus 260428 ExponentialODipEqual dedupmonthsplit cboost10maxagep028 20 1 0.7 1e-9 scipy_DE"
# "Metapneumovirus 2604292 ExponentialODipEqual dedupmonthsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 2604293 ExponentialODipEqual dedupmonthsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 2604294 ExponentialODipEqual dedupmonthsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 2604295 ExponentialODipEqual dedupmonthsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260429 ExponentialODipEqual dedupsplit cboost10maxagep028 1e-9 20 1 0.7 scipy_DE"
"RSV 260429 ExponentialODipEqual dedupmonthsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 260429 ExponentialODipEqual dedupmonthsplit maxagep05 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 260429 ExponentialODipEqual dedupmonthsplit maxagep05 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260429 ExponentialODipEqual dedupmonthsplit maxagep05 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260429 ExponentialODipEqual dedupmonthsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 260429 ExponentialODipEqual dedupmonthsplit nrmaxagep05 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 260429 ExponentialODipEqual dedupmonthsplit nrmaxagep05 1e-9 20 1 0.7 scipy_DE"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
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