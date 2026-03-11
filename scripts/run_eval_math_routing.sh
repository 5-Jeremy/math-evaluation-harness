set -ex

export CUDA_VISIBLE_DEVICES=0,1,2,3
# NOTE: Check if the prompt type is correct
PROMPT_TYPE="torl_deepmath_qwen"
MODEL_NAMES=(
    /home/codeGen_curriculum/checkpoints/torl_orig_dry_run/deepmath_baseline/global_step_660/actor/huggingface
)
# DATA_NAMES="minerva_math"
DATA_NAMES="gsm8k,math500,aime24"
SPLIT="test"
NUM_TEST_SAMPLE=-1

for MODEL_NAME_OR_PATH in "${MODEL_NAMES[@]}"; do
    echo "Evaluating model: ${MODEL_NAME_OR_PATH}"
    OUTPUT_DIR=results/${MODEL_NAME_OR_PATH}
    
    # single-gpu
    TOKENIZERS_PARALLELISM=false \
    python3 -u math_eval_with_routing.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --data_names ${DATA_NAMES} \
        --split ${SPLIT} \
        --prompt_type ${PROMPT_TYPE} \
        --num_test_sample ${NUM_TEST_SAMPLE} \
        --seed 0 \
        --temperature 0 \
        --n_sampling 1 \
        --top_p 1 \
        --start 0 \
        --end -1 \
        --save_outputs \
        --max_tokens_per_call 3072 \
        --use_vllm \
        --max_func_call 4 \
        --overwrite \
        2>&1 | tee "logs_${MODEL_NAME_OR_PATH//\//_}.log"
done


