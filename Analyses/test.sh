combinations=(
"RSV 260223 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
# "RSV 2602172 FlexStepwise NA flexage 1e-9 20 1 0.7"
# "RSV 2602173 FlexStepwise NA flexage 1e-9 20 1 0.7"
# "InfluenzaA 260217 FlexStepwise NA flexage 1e-9 20 1 0.7"
# "InfluenzaA 2602172 FlexStepwise NA flexage 1e-9 20 1 0.7"
# "InfluenzaA 2602173 FlexStepwise NA flexage 1e-9 20 1 0.7"
# "InfluenzaB 260217 FlexStepwise NA flexage 1e-9 20 1 0.7"
# "InfluenzaB 2602172 FlexStepwise NA flexage 1e-9 20 1 0.7"
# "InfluenzaB 2602173 FlexStepwise NA flexage 1e-9 20 1 0.7"
# "Metapneumovirus 260217 FlexStepwise NA flexage 1e-9 20 1 0.7"
# "Metapneumovirus 2602172 FlexStepwise NA flexage 1e-9 20 1 0.7"
# "Metapneumovirus 2602173 FlexStepwise NA flexage 1e-9 20 1 0.7"
# "Adenovirus 260217 FlexStepwise NA flexage 1e-9 20 1 0.7"
# "Adenovirus 2602172 FlexStepwise NA flexage 1e-9 20 1 0.7"
# "Adenovirus 2602173 FlexStepwise NA flexage 1e-9 20 1 0.7"
# "Parainfluenza3 260217 FlexStepwise NA flexage 1e-9 20 1 0.7"
# "Parainfluenza3 2602172 FlexStepwise NA flexage 1e-9 20 1 0.7"
# "Parainfluenza3 2602173 FlexStepwise NA flexage 1e-9 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done