seed=1
dataset_name="mme"
question_file="./experiments/data/MME_Benchmark_release_version/mme_hallucination.jsonl"
image_folder="./experiments/data/MME_Benchmark_release_version"

# llava
# model="llava"
# model_path="/mnt/server8_hard1/donguk/NEURIPS2024/checkpoints/llava-v1.5-7b"

# instructblip
model="instructblip"
model_path=None

gpu=0
export CUDA_VISIBLE_DEVICES=${gpu}

use_ritual=False
use_vcd=False
use_m3id=True

ritual_alpha_pos=3.0
ritual_alpha_neg=1.0
ritual_beta=0.1

log_path="./logs/mme/${model}"

python ./experiments/eval/mme_${model}.py \
--seed ${seed} \
--model-path ${model_path} \
--question-file ${question_file} \
--image-folder ${image_folder} \
--answers-file ./experiments/output/${model}_${dataset_name}_answers_seed${seed}.jsonl \
--use_ritual ${use_ritual} \
--use_vcd ${use_vcd} \
--use_m3id ${use_m3id} \
--ritual_alpha_pos ${ritual_alpha_pos} \
--ritual_alpha_neg ${ritual_alpha_neg} \
--ritual_beta ${ritual_beta} \

python ./experiments/eval/convert_answer_to_mme.py \
--output_path ./experiments/output/${model}_${dataset_name}_answers_seed${seed}.jsonl \
--seed ${seed} \
--model ${model} \
--log_path ${log_path}

python ./experiments/eval/eval_mme.py \
--results_dir ${log_path}/mme_answers \
--log_path ${log_path}
