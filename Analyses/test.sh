combinations=(
"InfluenzaA 260415 Exponential split daycarep5maxagep05 1e-9 1000 1e-2 0.5"
"InfluenzaA 260415 Exponential split daycarep5maxagep05 1e-9 1000 1e-3 0.9"
"InfluenzaA 260415 Exponential split daycarep5maxagep05 1e-9 1000 1e-2 0.9"
"InfluenzaA 260415 Exponential split daycarep5maxagep05 1e-9 1000 1e-2 0.1"
"InfluenzaA 260415 Exponential split daycarep5maxagep05 1e-9 1000 1e-4 0.5"
"InfluenzaA 260415 Exponential split daycarep5maxagep05 1e-9 1000 1e-4 0.9"
"InfluenzaB 260415 Exponential split daycarep5maxagep05 1e-9 1000 1e-2 0.5"
"InfluenzaB 260415 Exponential split daycarep5maxagep05 1e-9 1000 1e-3 0.9"
"InfluenzaB 260415 Exponential split daycarep5maxagep05 1e-9 1000 1e-2 0.9"
"InfluenzaB 260415 Exponential split daycarep5maxagep05 1e-9 1000 1e-2 0.1"
"InfluenzaB 260415 Exponential split daycarep5maxagep05 1e-9 1000 1e-4 0.5"
"InfluenzaB 260415 Exponential split daycarep5maxagep05 1e-9 1000 1e-4 0.9"
"Metapneumovirus 260415 Exponential split daycarep5maxagep02 1e-9 1000 1e-2 0.5"
"Metapneumovirus 260415 Exponential split daycarep5maxagep02 1e-9 1000 1e-3 0.9"
"Metapneumovirus 260415 Exponential split daycarep5maxagep02 1e-9 1000 1e-2 0.9"
"Metapneumovirus 260415 Exponential split daycarep5maxagep02 1e-9 1000 1e-2 0.1"
"Metapneumovirus 260415 Exponential split daycarep5maxagep02 1e-9 1000 1e-4 0.5"
"Metapneumovirus 260415 Exponential split daycarep5maxagep02 1e-9 1000 1e-4 0.9"
"Adenovirus 260415 Exponential split daycarep5maxagep02 1e-9 1000 1e-2 0.5"
"Adenovirus 260415 Exponential split daycarep5maxagep02 1e-9 1000 1e-3 0.9"
"Adenovirus 260415 Exponential split daycarep5maxagep02 1e-9 1000 1e-2 0.9"
"Adenovirus 260415 Exponential split daycarep5maxagep02 1e-9 1000 1e-2 0.1"
"Adenovirus 260415 Exponential split daycarep5maxagep02 1e-9 1000 1e-4 0.5"
"Adenovirus 260415 Exponential split daycarep5maxagep02 1e-9 1000 1e-4 0.9"
"Parainfluenza3 260415 Exponential split daycarep5maxagep02 1e-9 1000 1e-2 0.5"
"Parainfluenza3 260415 Exponential split daycarep5maxagep02 1e-9 1000 1e-3 0.9"
"Parainfluenza3 260415 Exponential split daycarep5maxagep02 1e-9 1000 1e-2 0.9"
"Parainfluenza3 260415 Exponential split daycarep5maxagep02 1e-9 1000 1e-2 0.1"
"Parainfluenza3 260415 Exponential split daycarep5maxagep02 1e-9 1000 1e-4 0.5"
"Parainfluenza3 260415 Exponential split daycarep5maxagep02 1e-9 1000 1e-4 0.9"
"RSV 260415 Exponential split daycarep5maxagep028 1e-9 1000 1e-2 0.5"
"RSV 260415 Exponential split daycarep5maxagep028 1e-9 1000 1e-3 0.9"
"RSV 260415 Exponential split daycarep5maxagep028 1e-9 1000 1e-2 0.9"
"RSV 260415 Exponential split daycarep5maxagep028 1e-9 1000 1e-2 0.1"
"RSV 260415 Exponential split daycarep5maxagep028 1e-9 1000 1e-4 0.5"
"RSV 260415 Exponential split daycarep5maxagep028 1e-9 1000 1e-4 0.9"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/polish.py $combination
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