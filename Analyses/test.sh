combinations=(
"Metapneumovirus 250325 Mobility X X 0.01 15 1 0.7"
"Metapneumovirus 250325 FlexStepwise X X 0.01 15 1 0.7"
"Adenovirus 250325 Mobility X X 0.01 15 1 0.7"
"Adenovirus 250325 FlexStepwise X X 0.01 15 1 0.7"
"Parainfluenza3 250325 Mobility X X 0.01 15 1 0.7"
"Parainfluenza3 250325 FlexStepwise X X 0.01 15 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done