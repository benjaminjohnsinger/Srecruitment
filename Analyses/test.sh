combinations=(
"RSV 260408 Exponential NA flexagep017 1e-9 20 400 0.7"
"RSV 260408 Exponential NA 2023-05-01flexagep028 1e-9 20 400 0.7"
"RSV 260408 FlexStepwise old_incidence_data flexagep005 1e-9 20 400 0.7"
"RSV 260408 Exponential old_incidence_data flexagep005 1e-9 20 400 0.7"
"RSV 260408 FlexStepwise old_incidence_data 2023-05-01flexagep005 1e-9 20 400 0.7"
"RSV 260408 Exponential old_incidence_data 2023-05-01flexagep005 1e-9 20 400 0.7"
"RSV 260408 FlexStepwise old_incidence_data flexagep028 1e-9 20 400 0.7"
"RSV 260408 Exponential old_incidence_data flexagep028 1e-9 20 400 0.7"
"RSV 260408 FlexStepwise old_incidence_data 2023-05-01flexagep028 1e-9 20 400 0.7"
"RSV 260408 Exponential old_incidence_data 2023-05-01flexagep028 1e-9 20 400 0.7"
# "RSV 260408 Exponential combo flexagep028 1e-9 20 400 0.7"
# "RSV 260408 Exponential incidence_data nrflexagep028 1e-9 20 400 0.7"
# "RSV 260408 Exponential combo nrflexagep028 1e-9 20 400 0.7"
# "RSV 260408 Exponential NA flexagep03 1e-9 20 1e-3 0.5 optax 400"
# "InfluenzaA 260408 Exponential NA flexagep03 1e-9 20 1e-3 0.5 optax 400"
# "InfluenzaB 260408 Exponential NA flexagep03 1e-9 20 1e-3 0.5 optax 400"
# "Metapneumovirus 260408 Exponential NA flexagep03 1e-9 20 1e-3 0.5 optax 400"
# "Parainfluenza3 260408 Exponential NA flexagep03 1e-9 20 1e-3 0.5 optax 400"
# "Adenovirus 260408 Exponential NA flexagep03 1e-9 20 1e-3 0.5 optax 400"
# "RSV 260407 Exponential combo flexagep01 1e-9 20 400 0.7"
# "RSV 260407 Exponential peaks_and_times nrflexagep01 1e-9 20 400 0.7"
# "RSV 260407 Exponential combo nrflexagep01 1e-9 20 400 0.7"
# "InfluenzaA 260407 Exponential peaks_and_times flexagep01 1e-9 20 400 0.7"
# "InfluenzaB 260407 Exponential peaks_and_times flexagep01 1e-9 20 400 0.7"
# "Metapneumovirus 260407 Exponential peaks_and_times flexagep01 1e-9 20 400 0.7"
# "Parainfluenza3 260407 Exponential peaks_and_times flexagep01 1e-9 20 400 0.7"
# "Adenovirus 260407 Exponential peaks_and_times flexagep01 1e-9 20 400 0.7"
# "InfluenzaA 260407 Exponential combo flexagep01 1e-9 20 400 0.7"
# "InfluenzaB 260407 Exponential combo flexagep01 1e-9 20 400 0.7"
# "Metapneumovirus 260407 Exponential combo flexagep01 1e-9 20 400 0.7"
# "Parainfluenza3 260407 Exponential combo flexagep01 1e-9 20 400 0.7"
# "Adenovirus 260407 Exponential combo flexagep01 1e-9 20 400 0.7"
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