combinations=(
"Adenovirus 260527 ExponentialODipLinear dedupsplit betaboundp9maxagep004 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260527 ExponentialODipLinear dedupsplit betaboundp9maxagep0045 1e-9 20 1 0.7 scipy_DE"

# "Metapneumovirus 260527 ExponentialODipLinear dedupsplit maxagep0085 1e-9 20 1 0.7 scipy_DE"
# "Adenovirus 260527 ExponentialODipLinear dedupsplit maxagep0041 1e-9 20 1 0.7 scipy_DE"
# "Adenovirus 260527 ExponentialODipLinear dedupsplit maxagep0042 1e-9 20 1 0.7 scipy_DE"
# "Adenovirus 260527 ExponentialODipLinear dedupsplit maxagep0043 1e-9 20 1 0.7 scipy_DE"
# "Adenovirus 260527 ExponentialODipLinear dedupsplit maxagep0044 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260527 ExponentialODipLinear dedupsplit maxagep007 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260527 ExponentialODipLinear dedupsplit maxagep0075 1e-9 20 1 0.7 scipy_DE"

# "Metapneumovirus 260527 ExponentialODipLinear dedupsplit maxagep0087 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260527 ExponentialODipLinear dedupsplit maxagep0083 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260527 ExponentialODipLinear dedupsplit maxagep0088 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaA 260527 ExponentialODipLinear dedupsplit maxagep05 1e-9 20 1 0.7 scipy_DE"

# "Metapneumovirus 260527 ExponentialODipLinear dedupsplit maxagep0081 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260527 ExponentialODipLinear dedupsplit maxagep0082 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260527 ExponentialODipLinear dedupsplit maxagep0084 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260527 ExponentialODipLinear dedupsplit maxagep0086 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260527 ExponentialODipLinear dedupsplit maxagep0089 1e-9 20 1 0.7 scipy_DE"

# "Parainfluenza3 260527 ExponentialODipLinear dedupsplit maxagep005 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260527 ExponentialODipLinear dedupsplit maxagep002 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260527 ExponentialODipLinear dedupsplit maxagep003 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260527 ExponentialODipLinear dedupsplit maxagep004 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260527 ExponentialODipLinear dedupsplit maxagep006 1e-9 20 1 0.7 scipy_DE"
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