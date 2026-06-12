combinations=(
"RSV 260531 ExponentialODipLinear dedupsac maxagep028 1e-9 200 2000 0.7 emcee_median"
"Metapneumovirus 260603 ExponentialODipLinear dedupsac maxagep015 1e-9 200 2000 0.7 emcee_median"
"Adenovirus 260531 ExponentialODipLinear dedupsac maxagep003 1e-9 200 2000 0.7 emcee_median"
"Parainfluenza3 260602 ExponentialODipLinear dedupsac maxagep004 1e-9 200 2000 0.7 emcee_median"
"InfluenzaA 260531 ExponentialODipLinear dedupsac maxagep035 1e-9 200 2000 0.7 emcee_median"
"InfluenzaB 260531 ExponentialODipLinear dedupsac maxagep035 1e-9 200 2000 0.7 emcee_median"
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