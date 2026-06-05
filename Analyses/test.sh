combinations=(
"RSV 260604 ExponentialODipLinear kireldedupsac maxagep028 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260604 ExponentialODipLinear kireldedupsac maxagep015 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260604 ExponentialODipLinear kireldedupsac maxagep004 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260604 ExponentialODipLinear kireldedupsac maxagep003 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 260604 ExponentialODipLinear kireldedupsac maxagep035 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 260604 ExponentialODipLinear kireldedupsac maxagep035 1e-9 20 1 0.7 scipy_DE"

"RSV 260604 ExponentialODipLinear fullireldedupsac maxagep028 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260604 ExponentialODipLinear fullireldedupsac maxagep015 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260604 ExponentialODipLinear fullireldedupsac maxagep004 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260604 ExponentialODipLinear fullireldedupsac maxagep003 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 260604 ExponentialODipLinear fullireldedupsac maxagep035 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 260604 ExponentialODipLinear fullireldedupsac maxagep035 1e-9 20 1 0.7 scipy_DE"
)

# for combination in "${combinations[@]}"; do
#     /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/profile.py $combination
# done

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