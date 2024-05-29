#!/bin/bash

seed=42
dataset_name="coco" # coco | aokvqa | gqa
type="random" # random | popular | adversarial

# llava
model="llava"
model_path="/mnt/server8_hard1/donguk/NEURIPS2024/checkpoints/llava-v1.5-7b"

# instructblip
# model="instructblip"
# model_path=None

pope_path="/mnt/server8_hard1/donguk/NEURIPS2024/rips_multi/neurips2024/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json"
data_path="/mnt/server18_hard0/jhjang/LVLM/crg/data/${dataset_name}/val2014"
log_path="./logs"

use_ritual=False
use_vcd=False
use_m3id=False

ritual_alpha_pos=3.0
ritual_alpha_neg=1.0
ritual_beta=0.1

experiment_index=0

#####################################
# Run single experiment
#####################################
export CUDA_VISIBLE_DEVICES=0
torchrun --nnodes=1 --nproc_per_node=1 --master_port 1234 eval_bench/pope_eval_${model}.py \
--seed ${seed} \
--model_path ${model_path} \
--model_base ${model} \
--pope_path ${pope_path} \
--data_path ${data_path} \
--log_path ${log_path} \
--use_ritual ${use_ritual} \
--use_vcd ${use_vcd} \
--use_m3id ${use_m3id} \
--ritual_alpha_pos ${ritual_alpha_pos} \
--ritual_alpha_neg ${ritual_alpha_neg} \
--ritual_beta ${ritual_beta} \
--experiment_index ${experiment_index}