combinations=(
"RSV 260529 ExponentialODipLinear dedupsac maxagep028 1e-9 200 3000 0.7"
"InfluenzaA 260529 ExponentialODipLinear dedupsac maxagep04 1e-9 200 2000 0.7"
"Parainfluenza3 260529 ExponentialODipLinear dedupsac maxagep007 1e-9 200 3000 0.7"
"InfluenzaB 260529 ExponentialODipLinear dedupsac maxagep04 1e-9 200 3000 0.7"
"Adenovirus 260529 ExponentialODipLinear dedupsac maxagep003 1e-9 200 2000 0.7"
"Metapneumovirus 260529 ExponentialODipLinear dedupsac maxagep008 1e-9 200 3000 0.7"
# "Parainfluenza3 260529 ExponentialODipLinear dedupsac maxagep0075 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260529 ExponentialODipLinear dedupsac maxagep0065 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260529 ExponentialODipLinear dedupsac maxagep006 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260529 ExponentialODipLinear dedupsac maxagep005 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260529 ExponentialODipLinear dedupsac maxagep002 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260529 ExponentialODipLinear dedupsac maxagep003 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260529 ExponentialODipLinear dedupsac maxagep0071 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260529 ExponentialODipLinear dedupsac maxagep0072 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260529 ExponentialODipLinear dedupsac maxagep0073 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260529 ExponentialODipLinear dedupsac maxagep0074 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260529 ExponentialODipLinear dedupsac maxagep0076 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260529 ExponentialODipLinear dedupsac maxagep0077 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260529 ExponentialODipLinear dedupsac maxagep0078 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260529 ExponentialODipLinear dedupsac maxagep0079 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260529 ExponentialODipLinear dedupsac maxagep01 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260529 ExponentialODipLinear dedupsac maxagep02 1e-9 20 1 0.7 scipy_DE"


# "Parainfluenza3 2605292 ExponentialODipLinear dedupsac maxagep0075 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 2605292 ExponentialODipLinear dedupsac maxagep0065 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 2605292 ExponentialODipLinear dedupsac maxagep006 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 2605292 ExponentialODipLinear dedupsac maxagep005 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 2605292 ExponentialODipLinear dedupsac maxagep002 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 2605292 ExponentialODipLinear dedupsac maxagep003 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 2605292 ExponentialODipLinear dedupsac maxagep0071 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 2605292 ExponentialODipLinear dedupsac maxagep0072 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 2605292 ExponentialODipLinear dedupsac maxagep0073 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 2605292 ExponentialODipLinear dedupsac maxagep0074 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 2605292 ExponentialODipLinear dedupsac maxagep0076 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 2605292 ExponentialODipLinear dedupsac maxagep0077 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 2605292 ExponentialODipLinear dedupsac maxagep0078 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 2605292 ExponentialODipLinear dedupsac maxagep0079 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 2605292 ExponentialODipLinear dedupsac maxagep01 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 2605292 ExponentialODipLinear dedupsac maxagep02 1e-9 20 1 0.7 scipy_DE"
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