combinations=(
# "InfluenzaA 260505 ExponentialODipEqual dedupsplit maxagep03 1e-9 200 1e-9 1e-6"
# "RSV 260505 ExponentialODipEqual dedupsplit maxagep028 1e-9 200 1e-9 1e-6"
# "Metapneumovirus 260505 ExponentialODipEqual dedupsplit maxagep028 1e-9 200 1e-7 1e-5"
# "Parainfluenza3 260505 ExponentialODipEqual dedupsplit maxagep028 1e-9 200 1e-7 1e-5"
# "InfluenzaB 260505 ExponentialODipEqual dedupsplit maxagep03 1e-9 200 1e-7 1e-5"
# "Adenovirus 260505 ExponentialODipEqual dedupsplit maxagep03 1e-9 200 1e-7 1e-5"

"InfluenzaA 260505 ExponentialODipEqual dedupsplit maxagep03 1e-9 20 1 0.7"
"InfluenzaB 260505 ExponentialODipEqual dedupsplit maxagep03 1e-9 20 1 0.7"
"Adenovirus 260505 ExponentialODipEqual dedupsplit maxagep03 1e-9 20 1 0.7"
"RSV 260505 ExponentialODipEqual dedupsplit maxagep028 1e-9 20 1 0.7"
"Metapneumovirus 260505 ExponentialODipEqual dedupsplit maxagep028 1e-9 20 1 0.7"
"Parainfluenza3 260505 ExponentialODipEqual dedupsplit maxagep028 1e-9 20 1 0.7"

# "Adenovirus 260505 ExponentialODipEqual dedupsplit betaboundp5maxagep002 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260514 ExponentialODipEqual dedupsplit maxagep005 1e-9 20 1 0.7 evosax_DE"
# "Metapneumovirus 260514 ExponentialODipEqual dedupsplit maxagep007 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260514 ExponentialODipEqual dedupsplit maxagep009 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260514 ExponentialODipEqual dedupsplit maxagep006 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260514 ExponentialODipEqual dedupsplit maxagep008 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260514 ExponentialODipEqual dedupsplit maxagep004 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260514 ExponentialODipEqual dedupsplit maxagep003 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260514 ExponentialODipEqual dedupsplit maxagep002 1e-9 20 1 0.7 scipy_DE"
)

# for combination in "${combinations[@]}"; do
#     /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/polish.py $combination
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