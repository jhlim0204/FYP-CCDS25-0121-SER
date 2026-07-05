#!/bin/bash
# Exit immediately if any command fails
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SRCIPT_DIR:$PYTHONPATH"

export CC=$(which gcc)
export CXX=$(which g++)

# ==============================================================================
# CLI ARGUMENTS
# ==============================================================================
DATASET=${1:?"Error: Dataset ('iemocap' or 'msp') must be provided as Arg 1."}
INPUT_PATH=${2:?"Error: Input path (WAV or CSV) must be provided as Arg 2."}
WITH_HISTORY=${3:?"Error: WITH_HISTORY ('True' or 'False') must be provided as Arg 3."}
CHECKPOINT_DIR=${4:-"../checkpoints/${DATASET}_checkpoints"}
OUTPUT_DIR=${5:-"../inference_outputs"}

if [[ "$WITH_HISTORY" != "True" && "$WITH_HISTORY" != "False" ]]; then
    echo "Error: WITH_HISTORY must be exactly 'True' or 'False' (got '$WITH_HISTORY')."
    exit 1
fi

TEMP_DIR="$OUTPUT_DIR/temp_data"
FINAL_DIR="$OUTPUT_DIR/final_predictions"

mkdir -p "$TEMP_DIR" "$FINAL_DIR"

echo "========================================================================"
echo "Starting End-to-End SER Inference Pipeline"
echo "Input:          $INPUT_PATH"
echo "Dataset:        $DATASET"
echo "Output Dir:     $OUTPUT_DIR"
echo "========================================================================"

# ==============================================================================
# STEP 1: AUDIO PREPROCESSING (venv_audio)
# ==============================================================================
echo -e "\n---> [1/2] Activating audio environment and running feature extraction..."
source activate venv_audio

python preprocess_inference.py \
    --input_path "$INPUT_PATH" \
    --dataset "$DATASET" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --temp_dir "$TEMP_DIR" \
    --use_pred_gender \
    --use_asr_pred \
    --with_history $WITH_HISTORY

# ==============================================================================
# STEP 2: LLM CONTEXTUAL EVALUATION (venv_llm)
# ==============================================================================
echo -e "\n---> [2/2] Activating LLM environment and executing DeepSpeed evaluation..."
source activate venv_llm

# Set model context length matching main_llm.sh logic
if [ "$DATASET" == "iemocap" ]; then
    MAX_LENGTH=2500
    LORA_LR=1e-4
    LORA_DIM=16
    LORA_ALPHA=16
    LORA_DROPOUT_PROB=0.05
    MAX_MASK_PROB=0.0
    historical_window=8

elif [ "$DATASET" == "msp" ]; then
    MAX_LENGTH=2048
    LORA_LR=1e-5
    LORA_DIM=32
    LORA_ALPHA=32
    LORA_DROPOUT_PROB=0.1
    MAX_MASK_PROB=0.1
    historical_window=0

else
    echo "Error: Invalid dataset '$DATASET'. Must be 'iemocap' or 'msp'."
    exit 1
fi

PORT=26000
LLM_CHECKPOINT="$CHECKPOINT_DIR/LLM_checkpoint"
if [ ! -d "$LLM_CHECKPOINT" ]; then
    echo "Error: Checkpoint directory '$LLM_CHECKPOINT' does not exist."
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
deepspeed --master_port=${PORT} LLM_code/main.py \
    --dataset ${DATASET} \
    --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --data_dir ${TEMP_DIR} \
    --output_dir ${FINAL_DIR} \
    --max_length ${MAX_LENGTH} \
    --batch_size 8 \
    --deepspeed_config ./LLM_code/llm_data_utils/deepspeed_config.json \
    --eval_batch_size 8 \
    --lora True \
    --do_train False \
    --do_eval True \
    --zero_shot False \
    --lora_dim ${LORA_DIM} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT_PROB} \
    --max_mask_prob ${MAX_MASK_PROB} \
    --checkpoint_dir ${LLM_CHECKPOINT}

echo -e "\nEnd-to-End Inference completed successfully! Results written to: $FINAL_DIR"