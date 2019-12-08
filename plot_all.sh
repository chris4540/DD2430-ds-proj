#!/bin/bash

folders="
exp_cat
exp_siamcat_m2_1
exp_siamcat_m2_10
exp_siamcat_m2_100
exp_siamcat_m2_1000
exp_siamcos
exp_siamese_m2_1
exp_siamese_m2_10
exp_siamese_m2_100
exp_siamese_m2_1000
"

for f in $folders; do
    exp_folder=exp_results/$f
    echo "Processing : " $exp_folder
    # python plt_search_results.py $exp_folder
    # python plt_emb.py $exp_folder
    # python make_topk_acc.py $exp_folder
    python plt_topk_acc.py $exp_folder
done