combinations=(
"RSV 251103 FlexStepwise mimm flexage 1e-9 20 1 0.7"
"RSV 251103 FlexStepwise mimmwane flexage 1e-9 20 1 0.7"
"RSV 251103 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"RSV 251103 FlexStepwise maxmimmwane flexagep01 1e-9 20 1 0.7"
"RSV 2511032 FlexStepwise mimm flexage 1e-9 20 1 0.7"
"RSV 2511032 FlexStepwise mimmwane flexage 1e-9 20 1 0.7"
"RSV 2511032 FlexStepwise NA flexagep01 1e-9 20 1 0.7"
"RSV 2511032 FlexStepwise maxmimmwane flexagep01 1e-9 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done