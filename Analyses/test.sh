combinations=(
"RSV 251024 FlexStepwise NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 251024 FlexStepwise NA flexage 1e-9 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done