combinations=(
"Metapneumovirus 260521 ExponentialODipLinear dedupsplit maxagep005 1e-9 200 2000 0.7"
"Parainfluenza3 260521 ExponentialODipLinear dedupsplit maxagep004 1e-9 200 2000 0.7"
# "RSV 260521 ExponentialODipLinear dedupsplit maxagep028 1e-9 200 2000 0.7"
# "InfluenzaA 260521 ExponentialODipLinear dedupsplit maxagep03 1e-9 200 2000 0.7"
# "InfluenzaB 260521 ExponentialODipLinear dedupsplit maxagep03 1e-9 200 2000 0.7"
# "Adenovirus 260521 ExponentialODipLinear dedupsplit betaboundp5maxagep003 1e-9 200 2000 0.7"
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