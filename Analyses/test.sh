combinations=(
"RSV 260423 ExponentialODipEqual split maxagep028 1e-9 5000 1e-6 1e-6"
"RSV 260423 ExponentialODipEqual split maxagep028 1e-9 5000 1e-8 1e-6"
"RSV 260423 ExponentialODipEqual split maxagep028 1e-9 5000 1e-10 1e-7"

# "RSV 260505 ExponentialInOutODipEqual dedupsplit maxagep028 1e-9 10 1 0.7 scipy_DE"
# "Metapneumovirus 260505 ExponentialInOutODipEqual dedupsplit maxagep028 1e-9 10 1 0.7 scipy_DE"
# "Parainfluenza3 260505 ExponentialInOutODipEqual dedupsplit maxagep028 1e-9 10 1 0.7 scipy_DE"
# "InfluenzaA 260505 ExponentialInOutODipEqual dedupsplit maxagep03 1e-9 10 1 0.7 scipy_DE"
# "InfluenzaB 260505 ExponentialInOutODipEqual dedupsplit maxagep03 1e-9 10 1 0.7 scipy_DE"
# "Adenovirus 260505 ExponentialInOutODipEqual dedupsplit maxagep03 1e-9 10 1 0.7 scipy_DE"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/polish.py $combination
done

# for combination in "${combinations[@]}"; do
#     /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
#     if [[ $combination == *"optax" ]]; then
#         /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_optax.py $combination
#     else
#         /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
#     fi
# done

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