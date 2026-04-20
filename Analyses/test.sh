combinations=(
# "RSV 260417 ExponentialInOut maxmimmsplit daycarep5maxagep028 1e-9 200 100 0.7"
# "InfluenzaA 260417 ExponentialInOut maxmimmsplit daycarep5maxagep05 1e-9 200 100 0.7"
# "Metapneumovirus 260417 ExponentialInOut maxmimmsplit daycarep5maxagep028 1e-9 200 100 0.7"
# "InfluenzaB 260417 ExponentialInOut maxmimmsplit daycarep5maxagep05 1e-9 200 100 0.7"
# "Adenovirus 260417 ExponentialInOut maxmimmsplit daycarep5maxagep05 1e-9 200 100 0.7"
# "Parainfluenza3 260417 ExponentialInOut maxmimmsplit daycarep5maxagep028 1e-9 200 100 0.7"

# "RSV 260415 Exponential split daycarep5maxagep028 1e-9 200 2000 0.7"
"Metapneumovirus 260416 ExponentialInOut maxmimmsplit daycarep5maxagep02 1e-9 200 2000 0.7"
# "InfluenzaA 260415 Exponential split daycarep5maxagep05 1e-9 200 2000 0.7"
# "InfluenzaB 260415 Exponential split daycarep5maxagep05 1e-9 200 2000 0.7"
# "Adenovirus 260415 Exponential split daycarep5maxagep02 1e-9 200 2000 0.7"
# "Parainfluenza3 260415 Exponential split daycarep5maxagep02 1e-9 200 2000 0.7"
# "RSV 260415 Exponential splitmaxmimm daycarep5maxagep028 1e-9 200 2000 0.7"
# "RSV 260415 ExponentialByAge7 split daycarep5maxagep028 1e-9 200 2000 0.7"
# "RSV 260415 ExponentialByAge5 splitmimm daycarep5maxagep028 1e-9 200 2000 0.7"

# "Metapneumovirus 260403 Exponential NA flexagep01 1e-9"
# "Metapneumovirus 260410 Exponential NA flexagep02 1e-9"
# "Metapneumovirus 260410 Exponential NA flexagep03 1e-9"
# "Metapneumovirus 260410 Exponential NA flexagep04 1e-9"
# "Metapneumovirus 260410 Exponential NA flexagep05 1e-9"
# "Metapneumovirus 260413 Exponential NA daycareflexagep03 1e-9"
# "Metapneumovirus 260413 ExponentialByAge5 NA daycareflexagep03 1e-9"
# "Metapneumovirus 260414 Exponential split daycareflexagep03 1e-9"
# "Metapneumovirus 260414 Exponential NA daycarep5flexagep03 1e-9"
# "Metapneumovirus 260415 Exponential split daycarep5maxagep02 1e-9"
# "Metapneumovirus 260415 ExponentialByAge7 split daycarep5maxagep02 1e-9"
# "Metapneumovirus 260415 Exponential split daycarep5maxagep02 1e-9"
# "Metapneumovirus 260415 Exponential mimmsplit daycarep5maxagep02 1e-9"
# "Metapneumovirus 260415 RSV0415 split daycarep5maxagep02 1e-9"
# "Metapneumovirus 260415 Exponential detrendmaxmimmsplit daycarep5maxagep02 1e-9"
# "Metapneumovirus 260415 ExponentialByAge7 maxmimmsplit daycarep5maxagep02 1e-9"
# "Metapneumovirus 260417 ExponentialInOut maxmimmsplit daycarep5maxagep028 1e-9"
)

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