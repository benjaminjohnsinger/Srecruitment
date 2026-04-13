combinations=(
"RSV 260410 PolicyDates NA flexagep05 1e-9 20 1 0.7 scipy_DE"
"Metapneumovirus 260410 PolicyDates NA flexagep05 1e-9 20 1 0.7 scipy_DE"
"InfluenzaA 260410 PolicyDates NA flexagep05 1e-9 20 1 0.7 scipy_DE"
"InfluenzaB 260410 PolicyDates NA flexagep05 1e-9 20 1 0.7 scipy_DE"
"Adenovirus 260410 PolicyDates NA flexagep05 1e-9 20 1 0.7 scipy_DE"
"Parainfluenza3 260410 PolicyDates NA flexagep05 1e-9 20 1 0.7 scipy_DE"

# "InfluenzaA 260410 Exponential NA flexagep05 1e-9 20 1000 0.7 best_evosax_skip_resampling"
# "InfluenzaA 260410 Exponential NA flexagep04 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaA 260410 Exponential NA flexagep06 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaA 260410 Exponential NA flexagep08 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaA 260410 Exponential NA flexagep09 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaB 260410 Exponential NA flexagep03 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaB 260410 Exponential NA flexagep07 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaB 260410 Exponential NA flexagep02 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaB 260410 Exponential NA flexagep04 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260410 Exponential NA flexagep03 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260410 Exponential NA flexagep05 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260410 Exponential NA flexagep02 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260410 Exponential NA flexagep04 1e-9 20 1 0.7 scipy_DE"
# "Adenovirus 260410 Exponential NA flexagep03 1e-9 20 1 0.7 scipy_DE"
# "Adenovirus 260410 Exponential NA flexagep05 1e-9 20 1 0.7 scipy_DE"
# "Adenovirus 260410 Exponential NA flexagep02 1e-9 20 1 0.7 scipy_DE"
# "Adenovirus 260410 Exponential NA flexagep04 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260410 Exponential NA flexagep03 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260410 Exponential NA flexagep05 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260410 Exponential NA flexagep02 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260410 Exponential NA flexagep04 1e-9 20 1 0.7 scipy_DE"

# "Metapneumovirus 260410 RSV0409 NA flexagep05 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260410 RSV0409 NA flexagep05 1e-9 20 400 0.7 best_evosax_skip_resampling"
# "InfluenzaA 260410 RSV0409 NA flexagep05 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaB 260410 RSV0409 NA flexagep05 1e-9 20 1 0.7 scipy_DE"
# "Adenovirus 260410 RSV0409 NA flexagep05 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260410 RSV0409 NA flexagep05 1e-9 20 1 0.7 scipy_DE"

# "Metapneumovirus 260410 RSV0409 NA flexagep03 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaA 260410 RSV0409 NA flexagep03 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaB 260410 RSV0409 NA flexagep03 1e-9 20 1 0.7 scipy_DE"
# "Adenovirus 260410 RSV0409 NA flexagep03 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260410 RSV0409 NA flexagep03 1e-9 20 1 0.7 scipy_DE"

# "RSV 260410 FlexStepwise NA flexagep99 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaA 260410 FlexStepwise NA flexagep99 1e-9 20 1 0.7 scipy_DE"
# "InfluenzaB 260410 FlexStepwise NA flexagep99 1e-9 20 1 0.7 scipy_DE"
# "Metapneumovirus 260410 FlexStepwise NA flexagep99 1e-9 20 1 0.7 scipy_DE"
# "Adenovirus 260410 FlexStepwise NA flexagep99 1e-9 20 1 0.7 scipy_DE"
# "Parainfluenza3 260410 FlexStepwise NA flexagep99 1e-9 20 1 0.7 scipy_DE"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
    if [[ $combination == *"optax" ]]; then
        /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_optax.py $combination
    else
        /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
    fi
done