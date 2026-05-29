combinations=(
"Adenovirus 260528 ExponentialODipLinear dedupsac maxagep0035 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260528 ExponentialODipLinear dedupsac maxagep004 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260528 ExponentialODipLinear dedupsac maxagep0045 1e-9 20 1 0.7 scipy_DE"
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