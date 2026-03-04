combinations=(
"RSV 260303 FlexStepwise incidence_data flexagep01 1e-9 20 400 0.7"
"RSV 260303 FlexStepwise smoothedincidence_data flexagep01 1e-9 20 400 0.7"
"RSV 2603032 FlexStepwise incidence_data flexagep01 1e-9 20 400 0.7"
"RSV 2603032 FlexStepwise smoothedincidence_data flexagep01 1e-9 20 400 0.7"
"RSV 2603033 FlexStepwise incidence_data flexagep01 1e-9 20 400 0.7"
"RSV 2603033 FlexStepwise smoothedincidence_data flexagep01 1e-9 20 400 0.7"
"InfluenzaA 260303 FlexStepwise incidence_data flexagep01 1e-9 20 400 0.7"
"InfluenzaA 260303 FlexStepwise smoothedincidence_data flexagep01 1e-9 20 400 0.7"
"InfluenzaA 2603032 FlexStepwise incidence_data flexagep01 1e-9 20 400 0.7"
"InfluenzaA 2603032 FlexStepwise smoothedincidence_data flexagep01 1e-9 20 400 0.7"
"InfluenzaA 2603033 FlexStepwise incidence_data flexagep01 1e-9 20 400 0.7"
"InfluenzaA 2603033 FlexStepwise smoothedincidence_data flexagep01 1e-9 20 400 0.7"
"RSV 260303 FlexStepwise NA flexagep01 1e-9 20 400 0.7"
"RSV 2603032 FlexStepwise NA flexagep01 1e-9 20 400 0.7"
"RSV 2603033 FlexStepwise NA flexagep01 1e-9 20 400 0.7"
"InfluenzaA 260303 FlexStepwise NA flexagep01 1e-9 20 400 0.7"
"InfluenzaA 2603032 FlexStepwise NA flexagep01 1e-9 20 400 0.7"
"InfluenzaA 2603033 FlexStepwise NA flexagep01 1e-9 20 400 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
done