combinations=(
"RSV 260223 FlexStepwise NA flexagep01 1e-9 4 1 0.7"
"Metapneumovirus 260223 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"Adenovirus 260223 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"RSV 2602232 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"Metapneumovirus 2602232 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"Adenovirus 2602233 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"RSV 2602233 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"Metapneumovirus 2602233 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"Adenovirus 2602232 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"InfluenzaA 260223 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"InfluenzaA 2602232 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"InfluenzaA 2602233 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"Parainfluenza3 260223 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"Parainfluenza3 2602232 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"Parainfluenza3 2602233 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"InfluenzaB 260223 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"InfluenzaB 2602232 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"InfluenzaB 2602233 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"InfluenzaB 2602234 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"InfluenzaB 2602235 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"InfluenzaB 2602236 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"InfluenzaB 2602237 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"InfluenzaB 2602238 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"InfluenzaB 2602239 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done