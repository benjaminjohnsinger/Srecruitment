combinations=(
"RSV 260415 Exponential split daycarep5maxagep028 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260415 Exponential split daycarep5maxagep02 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 260415 Exponential split daycarep5maxagep05 1e-9 20 1 0.7 scipy_DE"

"RSV 260415 ExponentialByAge7 split daycarep5maxagep028 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260415 ExponentialByAge7 split daycarep5maxagep02 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 260415 ExponentialByAge7 split daycarep5maxagep05 1e-9 20 1 0.7 scipy_DE"

"InfluenzaB 260415 Exponential split daycarep5maxagep05 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260415 Exponential split daycarep5maxagep02 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260415 Exponential split daycarep5maxagep02 1e-9 20 1 0.7 scipy_DE"

"InfluenzaB 260415 ExponentialByAge7 split daycarep5maxagep05 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260415 ExponentialByAge7 split daycarep5maxagep02 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260415 ExponentialByAge7 split daycarep5maxagep02 1e-9 20 1 0.7 scipy_DE"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
    if [[ $combination == *"optax" ]]; then
        /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_optax.py $combination
    else
        /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
    fi
done