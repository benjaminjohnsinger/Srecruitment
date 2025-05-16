combinations=(
"Adenovirus 2505142 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Adenovirus 2505143 FlexStepwise setimport flexage 0.01 20 1 0.7"
"Adenovirus 2505144 FlexStepwise setimport flexage 0.01 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done