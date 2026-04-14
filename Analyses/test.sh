combinations=(
"RSV 260413 Exponential NA daycareflexagep03 1e-9 20 1 0.7 scipy_DE"
"RSV 260413 ExponentialByAge4 NA daycareflexagep03 1e-9 20 1 0.7 scipy_DE"
"RSV 260413 ExponentialByAge5 NA daycareflexagep03 1e-9 20 1 0.7 scipy_DE"
"RSV 260413 ExponentialByAge6 NA daycareflexagep03 1e-9 20 1 0.7 scipy_DE"
"RSV 260413 ExponentialByAge4 NA flexagep03 1e-9 20 1 0.7 scipy_DE"
"RSV 260413 ExponentialByAge5 NA flexagep03 1e-9 20 1 0.7 scipy_DE"
"RSV 260413 ExponentialByAge6 NA flexagep03 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260413 Exponential NA daycareflexagep03 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 260413 Exponential NA daycareflexagep05 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 260413 Exponential NA daycareflexagep05 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260413 Exponential NA daycareflexagep04 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260413 Exponential NA daycareflexagep02 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260413 ExponentialByAge5 NA daycareflexagep03 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 260413 ExponentialByAge5 NA daycareflexagep05 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 260413 ExponentialByAge5 NA daycareflexagep05 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260413 ExponentialByAge5 NA daycareflexagep04 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260413 ExponentialByAge5 NA daycareflexagep02 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260413 ExponentialByAge5 NA flexagep03 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 260413 ExponentialByAge5 NA flexagep05 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 260413 ExponentialByAge5 NA flexagep05 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260413 ExponentialByAge5 NA flexagep04 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260413 ExponentialByAge5 NA flexagep02 1e-9 20 1 0.7 scipy_DE"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
    if [[ $combination == *"optax" ]]; then
        /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_optax.py $combination
    else
        /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
    fi
done