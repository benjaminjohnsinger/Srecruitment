combinations=(
"RSV 260406 Exponential peaks_and_times flexagep01 1e-9 20 400 0.7"
"RSV 260406 Exponential combo flexagep01 1e-9 20 400 0.7"
"RSV 260406 Exponential peaks_and_times nrflexagep01 1e-9 20 400 0.7"
"RSV 260406 Exponential combo nrflexagep01 1e-9 20 400 0.7"
"InfluenzaA 260406 Exponential peaks_and_times flexagep01 1e-9 20 400 0.7"
"InfluenzaB 260406 Exponential peaks_and_times flexagep01 1e-9 20 400 0.7"
"Metapneumovirus 260406 Exponential peaks_and_times flexagep01 1e-9 20 400 0.7"
"Parainfluenza3 260406 Exponential peaks_and_times flexagep01 1e-9 20 400 0.7"
"Adenovirus 260406 Exponential peaks_and_times flexagep01 1e-9 20 400 0.7"
"InfluenzaA 260406 Exponential combo flexagep01 1e-9 20 400 0.7"
"InfluenzaB 260406 Exponential combo flexagep01 1e-9 20 400 0.7"
"Metapneumovirus 260406 Exponential combo flexagep01 1e-9 20 400 0.7"
"Parainfluenza3 260406 Exponential combo flexagep01 1e-9 20 400 0.7"
"Adenovirus 260406 Exponential combo flexagep01 1e-9 20 400 0.7"
# "RSV 260406 Exponential NA nrflexagep01 1e-9 10 100 0.7"
# "RSV 260406 Exponential NA nrflexagep01 1e-9 20 400 0.7"
# "InfluenzaA 260406 Exponential NA nrflexagep01 1e-9 20 400 0.7"
# "InfluenzaB 260406 Exponential NA nrflexagep01 1e-9 20 400 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
    if [[ $combination == *"optax" ]]; then
        /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_optax.py $combination
    else
        /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
    fi
done