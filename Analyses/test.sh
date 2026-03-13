combinations=(
# "RSV 2603124 Exponential NA flexagep01 1e-9 20 400 0.7 evosax_skip_resampling"
# "Metapneumovirus 2603124 Exponential NA flexagep01 1e-9 20 400 0.1"
"Metapneumovirus 2603125 Exponential NA flexagep01 1e-9 20 400 0.9"
"Metapneumovirus 2603124 Exponential NA flexagep01 1e-9 20 400 0.1 evosax_skip_resampling"
"Metapneumovirus 2603125 Exponential NA flexagep01 1e-9 20 400 0.9 evosax_skip_resampling"
"RSV 2603126 Exponential NA flexagep01 1e-9 20 1000 0.7 evosax_skip_resampling"
"RSV 2603127 Exponential NA flexagep01 1e-9 100 1000 0.7 evosax_skip_resampling"
"RSV 2603128 Exponential NA flexagep01 1e-9 20 1000 0.9 evosax_skip_resampling"
"RSV 2603129 Exponential NA flexagep01 1e-9 100 1000 0.9 evosax_skip_resampling"
"RSV 26031210 Exponential NA flexagep01 1e-9 20 1000 0.5 evosax_skip_resampling"
"RSV 26031211 Exponential NA flexagep01 1e-9 100 1000 0.5 evosax_skip_resampling"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
done