combinations=(
"RSV 260409 FlexStepwise NA flexagep05 1e-9 20 1000 0.7"
"InfluenzaA 260409 FlexStepwise NA flexagep1 1e-9 20 1000 0.7"
"InfluenzaB 260409 FlexStepwise NA flexagep05 1e-9 20 1000 0.7"
"Metapneumovirus 260409 FlexStepwise NA flexagep05 1e-9 20 1000 0.7"
"Adenovirus 260409 FlexStepwise NA flexagep05 1e-9 20 1000 0.7"
"Parainfluenza3 260409 FlexStepwise NA flexagep05 1e-9 20 1000 0.7"

"RSV 260409 FlexStepwise NA flexagep1 1e-9 20 1000 0.7"
"InfluenzaA 260409 FlexStepwise NA flexagep05 1e-9 20 1000 0.7"
"InfluenzaB 260409 FlexStepwise NA flexagep1 1e-9 20 1000 0.7"
"Metapneumovirus 260409 FlexStepwise NA flexagep1 1e-9 20 1000 0.7"
"Adenovirus 260409 FlexStepwise NA flexagep1 1e-9 20 1000 0.7"
"Parainfluenza3 260409 FlexStepwise NA flexagep1 1e-9 20 1000 0.7"

"RSV 260409 FlexStepwise NA flexagep99 1e-9 20 1000 0.7"
"InfluenzaA 260409 FlexStepwise NA flexagep99 1e-9 20 1000 0.7"
"InfluenzaB 260409 FlexStepwise NA flexagep99 1e-9 20 1000 0.7"
"Metapneumovirus 260409 FlexStepwise NA flexagep99 1e-9 20 1000 0.7"
"Adenovirus 260409 FlexStepwise NA flexagep99 1e-9 20 1000 0.7"
"Parainfluenza3 260409 FlexStepwise NA flexagep99 1e-9 20 1000 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
    if [[ $combination == *"optax" ]]; then
        /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_optax.py $combination
    else
        /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
    fi
done