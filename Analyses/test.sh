combinations=(
"Metapneumovirus 260531 ExponentialODipLinear dedupsac maxagep006 1e-9 200 3000 0.7"
"Adenovirus 260531 ExponentialODipLinear dedupsac maxagep003 1e-9 200 2000 0.7"
"Parainfluenza3 260531 ExponentialODipLinear dedupsac maxagep005 1e-9 200 3000 0.7"
"RSV 260531 ExponentialODipLinear dedupsac maxagep028 1e-9 200 3000 0.7"
"Parainfluenza3 260531 ExponentialODipLinear dedupsac maxagep008 1e-9 200 2000 0.7"
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