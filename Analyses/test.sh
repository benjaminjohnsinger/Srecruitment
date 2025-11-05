combinations=(
"RSVfree 251104 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
"RSVfree 251104 FlexStepwise NA flexage 1e-9 20 1 0.7"
"RSVfree 2511042 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
"RSVfree 2511042 FlexStepwise NA flexage 1e-9 20 1 0.7"
"RSV 251103 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
"RSV 251103 FlexStepwise NA flexage 1e-9 20 1 0.7"
"RSV 2511032 FlexStepwise maxmimmwane flexage 1e-9 20 1 0.7"
"RSV 2511032 FlexStepwise NA flexage 1e-9 20 1 0.7"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
done