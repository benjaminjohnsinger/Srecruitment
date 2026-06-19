combinations=(
# "Adenovirus 260612 ExponentialODipp25 dedupsac maxagep003 1e-9 20 1 0.7 emcee_median"
# "InfluenzaA 260612 ExponentialODipp25 dedupsac maxagep035 1e-9 20 1 0.7 emcee_median"
# "RSV 260612 ExponentialODipp25 dedupsac maxagep028 1e-9 20 1 0.7 emcee_median"
# "Metapneumovirus 260612 ExponentialODipp25 dedupsac maxagep015 1e-9 20 1 0.7 emcee_median"
# "Parainfluenza3 260612 ExponentialODipp25 dedupsac maxagep004 1e-9 20 1 0.7 emcee_median"
# "InfluenzaB 260612 ExponentialODipp25 dedupsac maxagep035 1e-9 20 1 0.7 emcee_median"

"Metapneumovirus 260617 ExponentialODipp25 dedupsac fixage1maxagep005 1e-9 200 2000 0.7"
"Metapneumovirus 260617 ExponentialODipp25 dedupsac fixage1maxagep01 1e-9 200 2000 0.7"
"Metapneumovirus 260617 ExponentialODipp25 dedupsac fixage1maxagep015 1e-9 200 2000 0.7"
"Metapneumovirus 260617 ExponentialODipp25 dedupsac fixage1maxagep028 1e-9 200 2000 0.7"
"Metapneumovirus 260617 ExponentialODipp25 dedupsac fixage0maxagep005 1e-9 200 2000 0.7"
"Metapneumovirus 260617 ExponentialODipp25 dedupsac fixage0maxagep01 1e-9 200 2000 0.7"
"Metapneumovirus 260617 ExponentialODipp25 dedupsac fixage0maxagep015 1e-9 200 2000 0.7"
"Metapneumovirus 260617 ExponentialODipp25 dedupsac fixage0maxagep028 1e-9 200 2000 0.7"
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