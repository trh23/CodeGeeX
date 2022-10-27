# This script is used to test the inference of CodeGeeX.

GPU=$1
PROMPT_FILE=$2

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
MAIN_DIR=$(dirname "$SCRIPT_DIR")
TOKENIZER_PATH="$MAIN_DIR/codegeex/tokenizer/"

# import model configuration
source "$MAIN_DIR/configs/codegeex_13b.sh"

# export CUDA settings
if [ -z "$GPU" ]; then
  GPU=0
fi

# export CUDA_HOME=/usr/local/cuda-11.1/
export CUDA_VISIBLE_DEVICES=0,1,2,3

if [ -z "$PROMPT_FILE" ]; then
  PROMPT_FILE=$MAIN_DIR/tests/test_prompt.txt
fi

# remove --greedy if using sampling
CMD="python $MAIN_DIR/codegeex/benchmark/generate_samples.py \
        --prompt-file $PROMPT_FILE \
        --tokenizer-path $TOKENIZER_PATH \
        --micro-batch-size 10 \
        --out-seq-length 300 \
        --temperature 0.8 \
        --top-p 0.95 \
        --top-k 0 \
        --load-deepspeed \
        --gen-node-world-size 4 \
        --seed 42 \
        $MODEL_ARGS"

echo "$CMD"
eval "$CMD"
