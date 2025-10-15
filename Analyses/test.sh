combinations=(
"RSV 251014 Mobility NA flexage 1e-9 20 1 0.7"
"RSV 2510142 Mobility NA flexage 1e-9 20 1 0.7"
"RSV 251014 Mobility2 NA flexage 1e-9 20 1 0.7"
"RSV 2510142 Mobility2 NA flexage 1e-9 20 1 0.7"
"InfluenzaA 251014 Mobility NA flexage 1e-9 20 1 0.7"
"InfluenzaA 2510142 Mobility NA flexage 1e-9 20 1 0.7"
"InfluenzaA 251014 Mobility2 NA flexage 1e-9 20 1 0.7"
"InfluenzaA 2510142 Mobility2 NA flexage 1e-9 20 1 0.7"
"InfluenzaB 251014 Mobility NA flexage 1e-9 20 1 0.7"
"InfluenzaB 2510142 Mobility NA flexage 1e-9 20 1 0.7"
"InfluenzaB 251014 Mobility2 NA flexage 1e-9 20 1 0.7"
"InfluenzaB 2510142 Mobility2 NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 251014 Mobility NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2510142 Mobility NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 251014 Mobility2 NA flexage 1e-9 20 1 0.7"
"Metapneumovirus 2510142 Mobility2 NA flexage 1e-9 20 1 0.7"
"Adenovirus 251014 Mobility NA flexage 1e-9 20 1 0.7"
"Adenovirus 2510142 Mobility NA flexage 1e-9 20 1 0.7"
"Adenovirus 251014 Mobility2 NA flexage 1e-9 20 1 0.7"
"Adenovirus 2510142 Mobility2 NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 251014 Mobility NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2510142 Mobility NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 251014 Mobility2 NA flexage 1e-9 20 1 0.7"
"Parainfluenza3 2510142 Mobility2 NA flexage 1e-9 20 1 0.7"
"test 251014 FlexStepwise NA flexage 1e-9 20 1 0.7"
"test 2510142 FlexStepwise NA flexage 1e-9 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/BSinger/Documents/Srecruitment/.venv/bin/python /Users/BSinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done