combinations=(
"Metapneumovirus 250908 Mobility NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2509082 Mobility NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 250908 Mobility2 NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2509082 Mobility2 NA flexage 1e-9 20 1 0.7"
"Adenovirus 250908 Mobility NA flexage 1e-9 20 1 0.7"
"Adenovirus 2509082 Mobility NA flexage 1e-9 20 1 0.7"
"Adenovirus 250908 Mobility2 NA flexage 1e-9 20 1 0.7"
"Adenovirus 2509082 Mobility2 NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 250908 Mobility NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2509082 Mobility NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 250908 Mobility2 NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2509082 Mobility2 NA flexage 1e-9 20 1 0.7"
"RSV 250908 Mobility NA flexage 1e-9 20 1 0.7"
"RSV 2509082 Mobility NA flexage 1e-9 20 1 0.7"
"RSV 250908 Mobility2 NA flexage 1e-9 20 1 0.7"
"RSV 2509082 Mobility2 NA flexage 1e-9 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done