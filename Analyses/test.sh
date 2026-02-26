combinations=(
"RSV 260225 FlexStepwise NA ppflexage 1e-9 1000 0.003 3"
"InfluenzaA 260225 FlexStepwise NA ppflexage 1e-9 1000 0.003 3"
"InfluenzaB 260225 FlexStepwise NA ppflexage 1e-9 1000 0.003 3"
"Metapneumovirus 260225 FlexStepwise NA ppflexage 1e-9 1000 0.003 3"
"Adenovirus 260225 FlexStepwise NA ppflexage 1e-9 1000 0.003 3"
"Parainfluenza3 260225 FlexStepwise NA ppflexage 1e-9 1000 0.003 3"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
done