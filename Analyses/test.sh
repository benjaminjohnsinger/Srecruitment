combinations=(
# "Metapneumovirus 260421 FlexStepwise split maxagep028 1e-9"
# "Metapneumovirus 260421 ExponentialODipTune split maxagep028 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260421 Exponential ppsplit maxagep028 1e-9 20 1 0.7 scipy_DE"
# "RSV 260421 ExponentialODipTune split maxagep028 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260421 Exponential peaks_and_times_split maxagep028 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260421 Exponential2 split maxagep028 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260421 Mobility2 split maxagep028 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260421 PolicyDates split maxagep028 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 260421 ExponentialODipTune split maxagep05 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 260421 ExponentialODipTune split maxagep05 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260421 ExponentialODipTune split maxagep05 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260421 ExponentialODipTune split maxagep028 1e-9 20 1 0.7 scipy_DE"
"RSV 260421 Mobility split maxagep028 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 260421 Mobility split maxagep05 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 260421 Mobility split maxagep05 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260421 Mobility split maxagep05 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260421 Mobility split maxagep028 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260421 Mobility split maxagep028 1e-9 20 1 0.7 scipy_DE"
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