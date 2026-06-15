combinations=(
# "Adenovirus 260612 ExponentialODipp25 dedupsac maxagep003 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaA 260612 ExponentialODipp25 dedupsac maxagep035 1e-9 20 1 0.7 scipy_DE"
# "RSV 260612 ExponentialODipp25 dedupsac maxagep028 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260612 ExponentialODipp25 dedupsac maxagep015 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260612 ExponentialODipp25 dedupsac maxagep004 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaB 260612 ExponentialODipp25 dedupsac maxagep035 1e-9 20 1 0.7 scipy_DE"

# "Adenovirus 260612 ExponentialODipp5 dedupsac maxagep003 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaA 260612 ExponentialODipp5 dedupsac maxagep035 1e-9 20 1 0.7 scipy_DE"
# "RSV 260612 ExponentialODipp5 dedupsac maxagep028 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260612 ExponentialODipp5 dedupsac maxagep015 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260612 ExponentialODipp5 dedupsac maxagep004 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaB 260612 ExponentialODipp5 dedupsac maxagep035 1e-9 20 1 0.7 scipy_DE"

# "Adenovirus 260612 ExponentialODipp75 dedupsac maxagep003 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaA 260612 ExponentialODipp75 dedupsac maxagep035 1e-9 20 1 0.7 scipy_DE"
# "RSV 260612 ExponentialODipp75 dedupsac maxagep028 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260612 ExponentialODipp75 dedupsac maxagep015 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260612 ExponentialODipp75 dedupsac maxagep004 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaB 260612 ExponentialODipp75 dedupsac maxagep035 1e-9 20 1 0.7 scipy_DE"

# "Adenovirus 260612 ExponentialODipp25 dedupsac maxagep004 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260612 ExponentialODipp25 dedupsac maxagep02 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260612 ExponentialODipp25 dedupsac maxagep005 1e-9 20 1 0.7 scipy_DE"

# "Adenovirus 260612 ExponentialODipp25 dedupsac maxagep002 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260612 ExponentialODipp25 dedupsac maxagep01 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260612 ExponentialODipp25 dedupsac maxagep003 1e-9 20 1 0.7 scipy_DE"

"Adenovirus 260612 ExponentialODipp5 dedupsac maxagep003 1e-9 200 2000 0.7"
"InfluenzaA 260612 ExponentialODipp5 dedupsac maxagep035 1e-9 200 2000 0.7"
"RSV 260612 ExponentialODipp5 dedupsac maxagep028 1e-9 200 2000 0.7"
"Metapneumovirus 260612 ExponentialODipp5 dedupsac maxagep015 1e-9 200 2000 0.7"
"Parainfluenza3 260612 ExponentialODipp5 dedupsac maxagep004 1e-9 200 2000 0.7"
"InfluenzaB 260612 ExponentialODipp5 dedupsac maxagep035 1e-9 200 2000 0.7"
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