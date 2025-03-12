combinations=(
"InfluenzaA 250310 Mobility nb nr 15 1 0.7"
"InfluenzaA 250310 FlexStepwise nb nr 15 1 0.7"
"InfluenzaB 250310 Mobility nb nr 15 1 0.7"
"InfluenzaB 250310 FlexStepwise nb nr 15 1 0.7"
"InfluenzaA 250310 Mobility nb X 15 1 0.7"
"InfluenzaA 250310 FlexStepwise nb X 15 1 0.7"
"InfluenzaB 250310 Mobility nb X 15 1 0.7"
"InfluenzaB 250310 FlexStepwise nb X 15 1 0.7"
"RSV 250310 Mobility nb X 15 1 0.7"
"RSV 250310 FlexStepwise nb X 15 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done

/Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py RSV 250310 FlexStepwise nb X 15 1 0.7