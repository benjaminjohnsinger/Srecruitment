combinations=(
"Metapneumovirus 260420 Exponential split daycarep5maxagep01 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260420 Exponential split daycarep5maxagep028 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260420 Exponential split daycarep5maxagep04 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260420 Exponential split daycarep5maxagep05 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260420 Exponential split daycarep5maxagep01 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260420 Exponential split daycarep5maxagep028 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260420 Exponential split daycarep5maxagep04 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260420 Exponential split daycarep5maxagep01 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260420 Exponential split daycarep5maxagep028 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260420 Exponential split daycarep5maxagep04 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260420 Exponential split daycarep5maxagep05 1e-9 20 1 0.7 scipy_DE"
# "RSV 260417 ExponentialInOut maxmimmsplit daycarep5maxagep028 1e-9 200 100 0.7"
# "InfluenzaA 260417 ExponentialInOut maxmimmsplit daycarep5maxagep05 1e-9 200 100 0.7"
# "Metapneumovirus 260417 ExponentialInOut maxmimmsplit daycarep5maxagep028 1e-9 200 100 0.7"
# "InfluenzaB 260417 ExponentialInOut maxmimmsplit daycarep5maxagep05 1e-9 200 100 0.7"
# "Adenovirus 260417 ExponentialInOut maxmimmsplit daycarep5maxagep05 1e-9 200 100 0.7"
# "Parainfluenza3 260417 ExponentialInOut maxmimmsplit daycarep5maxagep028 1e-9 200 100 0.7"
)

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