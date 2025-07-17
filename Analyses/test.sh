combinations=(
"RSV 2507092 2507153 0.005 flexage 0.01 20 1 0.7"
"InfluenzaA 2507092 2507153 0.005 flexage 0.01 20 1 0.7"
"InfluenzaB 250709 2507153 0.005 flexage 0.01 20 1 0.7"
"Metapneumovirus 2507092 2507153 0.005 flexage 0.01 20 1 0.7"
"Adenovirus 2506252 2507153 0.005 flexage 0.01 20 1 0.7"
"Parainfluenza3 2507092 2507153 0.005 flexage 0.01 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done