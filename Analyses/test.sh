combinations=(
# "Metapneumovirus 260428 ExponentialODipEqual NA cboost2maxagep028 20 1 0.7 1e-9 scipy_DE"
# "Metapneumovirus 260428 ExponentialODipEqual monthsplit cboost10maxagep028 20 1 0.7 1e-9 scipy_DE"
"RSV 260428 ExponentialODipEqual irelmonthsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
"RSV 260428 ExponentialODipEqual monthsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260428 ExponentialODipEqual monthsplit maxagep05 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260428 ExponentialODipEqual monthsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 260428 ExponentialODipEqual monthsplit nrmaxagep05 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 260428 ExponentialODipEqual monthsplit nrmaxagep05 1e-9 20 1 0.7 scipy_DE"
"RSV 2604282 ExponentialODipEqual monthsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 260428 ExponentialODipEqual monthsplit maxagep05 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 260428 ExponentialODipEqual monthsplit maxagep05 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 2604282 ExponentialODipEqual monthsplit maxagep05 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 2604282 ExponentialODipEqual monthsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 2604282 ExponentialODipEqual monthsplit nrmaxagep05 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 2604282 ExponentialODipEqual monthsplit nrmaxagep05 1e-9 20 1 0.7 scipy_DE"
"RSV 2604283 ExponentialODipEqual monthsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 2604283 ExponentialODipEqual monthsplit maxagep05 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 2604283 ExponentialODipEqual monthsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 2604283 ExponentialODipEqual monthsplit nrmaxagep05 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 2604283 ExponentialODipEqual monthsplit nrmaxagep05 1e-9 20 1 0.7 scipy_DE"
"RSV 2604284 ExponentialODipEqual monthsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 2604284 ExponentialODipEqual monthsplit maxagep05 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 2604284 ExponentialODipEqual monthsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 2604284 ExponentialODipEqual monthsplit nrmaxagep05 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 2604284 ExponentialODipEqual monthsplit nrmaxagep05 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 2604274 ExponentialODipEqual monthsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 2604275 ExponentialODipEqual monthsplit maxagep028 1e-9 40 1 0.7 scipy_DE"
# "Metapneumovirus 2604276 ExponentialODipEqual monthsplit maxagep028 1e-9 40 1 0.7 scipy_DE"
# "Metapneumovirus 2604277 ExponentialODipEqual monthsplit maxagep028 1e-9 80 1 0.7 scipy_DE"
# "Metapneumovirus 2604278 ExponentialODipEqual monthsplit maxagep028 1e-9 80 1 0.7 scipy_DE"
# "Metapneumovirus 2604279 ExponentialODipEqual monthsplit maxagep028 1e-9 100 1 0.7 scipy_DE"
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