combinations=(
"RSV 260423 ExponentialODipEqual split maxagep028 1e-9 5000 1e-12 1e-9"
# "RSV 260423 ExponentialODipEqual split maxagep028 1e-9 5000 1e-5 1e-5"

"InfluenzaA 260424 ExponentialODipEqual split maxagep05 1e-9 5000 1e-4 1e-4"
"InfluenzaB 260424 ExponentialODipEqual split maxagep05 1e-9 5000 1e-4 1e-4"
"Metapneumovirus 260424 ExponentialODipEqual split maxagep028 1e-9 5000 1e-4 1e-4"
"Parainfluenza3 260423 ExponentialODipEqual split maxagep028 1e-9 5000 1e-4 1e-4"
"Adenovirus 260423 ExponentialODipEqual split maxagep05 1e-9 5000 1e-4 1e-4"

"InfluenzaA 260424 ExponentialODipEqual split maxagep05 1e-9 5000 1e-5 1e-5"
"InfluenzaB 260424 ExponentialODipEqual split maxagep05 1e-9 5000 1e-5 1e-5"
"Metapneumovirus 260424 ExponentialODipEqual split maxagep028 1e-9 5000 1e-5 1e-5"
"Parainfluenza3 260423 ExponentialODipEqual split maxagep028 1e-9 5000 1e-5 1e-5"
"Adenovirus 260423 ExponentialODipEqual split maxagep05 1e-9 5000 1e-5 1e-5"

"RSV 260423 ExponentialODipEqual split maxagep028 1e-9 5000 1e-12 1e-9"
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