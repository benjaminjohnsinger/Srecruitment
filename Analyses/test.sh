combinations=(
# "RSV 260521 ExponentialODipLinear dedupsplit maxagep028 1e-9 200 2000 0.7"
# "InfluenzaA 260521 ExponentialODipLinear dedupsplit maxagep03 1e-9 200 2000 0.7"
# "InfluenzaB 260521 ExponentialODipLinear dedupsplit maxagep03 1e-9 200 2000 0.7"
# "Metapneumovirus 260521 ExponentialODipLinear dedupsplit maxagep005 1e-9 200 2000 0.7"

"RSV 260505 ExponentialODipEqual dedupsplit maxagep028 BETA"
"RSV 260505 ExponentialODipEqual dedupsplit maxagep028 S_REL1"
"InfluenzaA 260505 ExponentialODipEqual dedupsplit maxagep03 BETA"
"InfluenzaA 260505 ExponentialODipEqual dedupsplit maxagep03 S_REL1"
"InfluenzaB 260505 ExponentialODipEqual dedupsplit maxagep03 BETA"
"InfluenzaB 260505 ExponentialODipEqual dedupsplit maxagep03 S_REL1"
"Metapneumovirus 260505 ExponentialODipEqual dedupsplit maxagep028 BETA"
"Metapneumovirus 260505 ExponentialODipEqual dedupsplit maxagep028 S_REL1"
"Parainfluenza3 260505 ExponentialODipEqual dedupsplit maxagep028 BETA"
"Parainfluenza3 260505 ExponentialODipEqual dedupsplit maxagep028 S_REL1"

"RSV 260505 ExponentialODipEqual dedupsplit maxagep028 S_REL2"
"RSV 260505 ExponentialODipEqual dedupsplit maxagep028 OFFSET"
"RSV 260505 ExponentialODipEqual dedupsplit maxagep028 SEASONALITY"
"RSV 260505 ExponentialODipEqual dedupsplit maxagep028 WANE2"
"RSV 260505 ExponentialODipEqual dedupsplit maxagep028 F1"
"RSV 260505 ExponentialODipEqual dedupsplit maxagep028 R1"
"RSV 260505 ExponentialODipEqual dedupsplit maxagep028 AGE_OBS_2"
"RSV 260505 ExponentialODipEqual dedupsplit maxagep028 AGE_OBS_3"
"RSV 260505 ExponentialODipEqual dedupsplit maxagep028 AGE_OBS_4"
"RSV 260505 ExponentialODipEqual dedupsplit maxagep028 AGE_OBS_5"
"RSV 260505 ExponentialODipEqual dedupsplit maxagep028 AGE_OBS_6"
"RSV 260505 ExponentialODipEqual dedupsplit maxagep028 AGE_OBS_7"
"RSV 260505 ExponentialODipEqual dedupsplit maxagep028 AGE_OBS_8"

"Metapneumovirus 260505 ExponentialODipEqual dedupsplit maxagep028 S_REL2"
"Metapneumovirus 260505 ExponentialODipEqual dedupsplit maxagep028 OFFSET"
"Metapneumovirus 260505 ExponentialODipEqual dedupsplit maxagep028 SEASONALITY"
"Metapneumovirus 260505 ExponentialODipEqual dedupsplit maxagep028 WANE2"
"Metapneumovirus 260505 ExponentialODipEqual dedupsplit maxagep028 D_REL1"
"Metapneumovirus 260505 ExponentialODipEqual dedupsplit maxagep028 D_REL2"
"Metapneumovirus 260505 ExponentialODipEqual dedupsplit maxagep028 F1"
"Metapneumovirus 260505 ExponentialODipEqual dedupsplit maxagep028 R1"
"Metapneumovirus 260505 ExponentialODipEqual dedupsplit maxagep028 AGE_OBS_1"
"Metapneumovirus 260505 ExponentialODipEqual dedupsplit maxagep028 AGE_OBS_2"
"Metapneumovirus 260505 ExponentialODipEqual dedupsplit maxagep028 AGE_OBS_3"
"Metapneumovirus 260505 ExponentialODipEqual dedupsplit maxagep028 AGE_OBS_4"
"Metapneumovirus 260505 ExponentialODipEqual dedupsplit maxagep028 AGE_OBS_5"
"Metapneumovirus 260505 ExponentialODipEqual dedupsplit maxagep028 AGE_OBS_6"
"Metapneumovirus 260505 ExponentialODipEqual dedupsplit maxagep028 AGE_OBS_7"
)

for combination in "${combinations[@]}"; do
    /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/profile.py $combination
done

# for combination in "${combinations[@]}"; do
#     # /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
#     if [[ $combination == *"optax" ]]; then
#         /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_optax.py $combination
#     else
#         /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
#     fi
# done

# for i in {1..100}; do
#     echo "Chunk $i"
#     for combination in "${combinations[@]}"; do
#         /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/fit_opt.py $combination
#         if [[ $combination == *"optax" ]]; then
#             /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_optax.py $combination
#         else
#             /Users/bjsinger/Documents/Srecruitment/.venv/bin/python /Users/bjsinger/Documents/Srecruitment/Analyses/plot_opt.py $combination
#         fi
#     done
# done