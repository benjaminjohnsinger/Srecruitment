combinations=(
"RSV 250430 X setimport 2020-03-19 0.01 15 1 0.7"
"InfluenzaA 250430 X setimport 2020-03-19 0.01 15 1 0.7"
"InfluenzaB 250430 X setimport 2020-03-19 0.01 15 1 0.7"
"Metapneumovirus 250430 X setimport 2020-03-19 0.01 15 1 0.7"
"Adenovirus 250430 X setimport 2020-03-19 0.01 15 1 0.7"
"Parainfluenza3 250430 X setimport 2020-03-19 0.01 15 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done