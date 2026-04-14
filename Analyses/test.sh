combinations=(
"RSV 260414 Exponential split daycareflexagep03 1e-9 20 1 0.7 scipy_DE"
"RSV 260414 Exponential NA daycareflexagep03 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260414 Exponential split daycareflexagep03 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 260414 Exponential split daycareflexagep05 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 260414 Exponential split daycareflexagep05 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260414 Exponential split daycareflexagep04 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260414 Exponential split daycareflexagep02 1e-9 20 1 0.7 scipy_DE"

"Metapneumovirus 260414 Exponential NA daycareflexagep03 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 260414 Exponential NA daycareflexagep05 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 260414 Exponential NA daycareflexagep05 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260414 Exponential NA daycareflexagep04 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260414 Exponential NA daycareflexagep02 1e-9 20 1 0.7 scipy_DE"

"RSV 260414 Exponential NA daycarep5flexagep03 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260414 Exponential NA daycarep5flexagep03 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 260414 Exponential NA daycarep5flexagep05 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 260414 Exponential NA daycarep5flexagep05 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260414 Exponential NA daycarep5flexagep04 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260414 Exponential NA daycarep5flexagep02 1e-9 20 1 0.7 scipy_DE"

"RSV 260414 ExponentialByAge5 NA daycarep5flexagep03 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260414 ExponentialByAge5 NA daycarep5flexagep03 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 260414 ExponentialByAge5 NA daycarep5flexagep05 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 260414 ExponentialByAge5 NA daycarep5flexagep05 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260414 ExponentialByAge5 NA daycarep5flexagep04 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260414 ExponentialByAge5 NA daycarep5flexagep02 1e-9 20 1 0.7 scipy_DE"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
    if [[ $combination == *"optax" ]]; then
        /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_optax.py $combination
    else
        /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
    fi
done