combinations=(
"RSV 250429 FlexStepwise setimport maternal 0.01 15 1 0.7"
"InfluenzaA 250429 FlexStepwise setimport maternal 0.01 15 1 0.7"
"Adenovirus 250429 FlexStepwise setimport maternal 0.01 15 1 0.7"
"Parainfluenza3 250429 FlexStepwise setimport maternal 0.01 15 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done