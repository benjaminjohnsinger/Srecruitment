combinations=(
"RSV 260605 ExponentialODipLinear kireldedupsac maxagep028 1e-9 200 2000 0.7"
"Metapneumovirus 260605 ExponentialODipLinear kireldedupsac maxagep015 1e-9 200 2000 0.7"
"Adenovirus 260605 ExponentialODipLinear kireldedupsac maxagep003 1e-9 200 2000 0.7"
"Parainfluenza3 260605 ExponentialODipLinear kireldedupsac maxagep004 1e-9 200 2000 0.7"
"InfluenzaA 260605 ExponentialODipLinear kireldedupsac maxagep035 1e-9 200 2000 0.7"
"InfluenzaB 260605 ExponentialODipLinear kireldedupsac maxagep035 1e-9 200 2000 0.7"

"RSV 260605 ExponentialODipLinear unimmlimdedupsac maxagep028 1e-9 200 2000 0.7"
"Metapneumovirus 260605 ExponentialODipLinear unimmlimdedupsac maxagep015 1e-9 200 2000 0.7"
"Adenovirus 260605 ExponentialODipLinear unimmlimdedupsac maxagep003 1e-9 200 2000 0.7"
"Parainfluenza3 260605 ExponentialODipLinear unimmlimdedupsac maxagep004 1e-9 200 2000 0.7"
"InfluenzaA 260605 ExponentialODipLinear unimmlimdedupsac maxagep035 1e-9 200 2000 0.7"
"InfluenzaB 260605 ExponentialODipLinear unimmlimdedupsac maxagep035 1e-9 200 2000 0.7"

"Metapneumovirus 260605 ExponentialODipLinear kireldedupsac maxagep01 1e-9 200 2000 0.7"
"Adenovirus 260605 ExponentialODipLinear kireldedupsac maxagep002 1e-9 200 2000 0.7"
"Parainfluenza3 260605 ExponentialODipLinear kireldedupsac maxagep003 1e-9 200 2000 0.7"
"InfluenzaA 260605 ExponentialODipLinear kireldedupsac maxagep03 1e-9 200 2000 0.7"
"InfluenzaB 260605 ExponentialODipLinear kireldedupsac maxagep03 1e-9 200 2000 0.7"

"Metapneumovirus 260605 ExponentialODipLinear kireldedupsac maxagep02 1e-9 200 2000 0.7"
"Adenovirus 260605 ExponentialODipLinear kireldedupsac maxagep004 1e-9 200 2000 0.7"
"Parainfluenza3 260605 ExponentialODipLinear kireldedupsac maxagep005 1e-9 200 2000 0.7"
"InfluenzaA 260605 ExponentialODipLinear kireldedupsac maxagep04 1e-9 200 2000 0.7"
"InfluenzaB 260605 ExponentialODipLinear kireldedupsac maxagep04 1e-9 200 2000 0.7"

"Metapneumovirus 260605 ExponentialODipLinear unimmlimdedupsac maxagep01 1e-9 200 2000 0.7"
"Adenovirus 260605 ExponentialODipLinear unimmlimdedupsac maxagep002 1e-9 200 2000 0.7"
"Parainfluenza3 260605 ExponentialODipLinear unimmlimdedupsac maxagep003 1e-9 200 2000 0.7"
"InfluenzaA 260605 ExponentialODipLinear unimmlimdedupsac maxagep03 1e-9 200 2000 0.7"
"InfluenzaB 260605 ExponentialODipLinear unimmlimdedupsac maxagep03 1e-9 200 2000 0.7"

"Metapneumovirus 260605 ExponentialODipLinear unimmlimdedupsac maxagep02 1e-9 200 2000 0.7"
"Adenovirus 260605 ExponentialODipLinear unimmlimdedupsac maxagep004 1e-9 200 2000 0.7"
"Parainfluenza3 260605 ExponentialODipLinear unimmlimdedupsac maxagep005 1e-9 200 2000 0.7"
"InfluenzaA 260605 ExponentialODipLinear unimmlimdedupsac maxagep04 1e-9 200 2000 0.7"
"InfluenzaB 260605 ExponentialODipLinear unimmlimdedupsac maxagep04 1e-9 200 2000 0.7"
)

# for combination in "${combinations[@]}"; do
#     /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/profile.py $combination
# done

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