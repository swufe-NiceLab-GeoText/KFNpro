#!/bin/bash
seed=171
dataset_name="oos"
task="5w5s"
output="first"
gpu=0
kshot=5
numf=0
dataset_num=("01" "02" "03" "04" "05")
for k in "${dataset_num[@]}";do
  python ../main2.py --config ./attack_oos_5shot.json \
  --seed $seed \
  --gpu $gpu \
  --kshot $kshot \
  --output $output \
  --numFreeze $numf \
  --dataset_num $k \
  --dataset_name $dataset_name \
  --task $task
done

