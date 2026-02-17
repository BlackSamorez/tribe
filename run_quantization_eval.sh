#!/bin/bash

# Create results file
AUTHOR=meta-llama
MODEL=Llama-3.2-1B
DECODE_MODE="3inst_fp4"
AQUANT="fp4_absmax"
GROUP_SIZE="16"
HADAMARD_SIZE="128"
EXTRA_WSCALING_SCHEME="None"

if [ "$EXTRA_WSCALING_SCHEME" == "None" ]; then
    EXTRA_APPENDIX=""
else
    EXTRA_APPENDIX=-w${EXTRA_WSCALING_SCHEME}
fi

RESULTS_FILE="results/${MODEL}-${DECODE_MODE}-${AQUANT}-gs${GROUP_SIZE}-hs${HADAMARD_SIZE}${EXTRA_APPENDIX}.txt"
echo "Bit,Dataset,PPL" > $RESULTS_FILE

# Function to run full pipeline for a given bit width
run_pipeline() {
    local bits=$1
    echo "Running pipeline for ${bits}-bit quantization..."

    local HESSIANS_PATH=~/hessians/${MODEL}-multihess
    local TORCH_CKPT=~/models/QTIP/${MODEL}-${bits}bit-${DECODE_MODE}-gs${GROUP_SIZE}-hs${HADAMARD_SIZE}${EXTRA_APPENDIX}
    local HF_CKPT=~/models/QTIP/${MODEL}-${bits}bit-${DECODE_MODE}-gs${GROUP_SIZE}-hs${HADAMARD_SIZE}${EXTRA_APPENDIX}-hf
    
    # Only do steps 1 and 2 if $HF_CKPT dir does not exist
    if [ ! -d "$HF_CKPT" ]; then
        # Step 1: Quantize
        echo "HF path ${HF_CKPT} does not exist. Re-quantizing..."

        echo "Step 1: Quantizing model to ${bits}-bit..."
        python -m quantize_llama.quantize_finetune_llama \
            --base_model ${AUTHOR}/${MODEL} \
            --in_hess_path $HESSIANS_PATH \
            --save_path $TORCH_CKPT \
            --group_size $GROUP_SIZE \
            --hadamard_size $HADAMARD_SIZE \
            --extra_wscaling_scheme $EXTRA_WSCALING_SCHEME \
            --L 16 --K $bits --V 1 --tlut_bits 0 --decode_mode $DECODE_MODE \
            --ft_epochs 0
        
        if [ $? -ne 0 ]; then
            echo "Error: Quantization failed for ${bits}-bit"
            return 1
        fi
        
        # Step 2: Convert to HF format
        echo "Step 2: Converting to HuggingFace format..."
        python -m quantize_llama.hfize_llama \
            --quantized_path $TORCH_CKPT \
            --hf_output_path $HF_CKPT
        
        if [ $? -ne 0 ]; then
            echo "Error: HF conversion failed for ${bits}-bit"
            return 1
        fi
    else
        echo "HF checkpoint at $HF_CKPT exists, skipping quantization and conversion steps."
    fi
    
    # Step 3: Evaluate and capture results
    echo "Step 3: Evaluating perplexity..."
    eval_output=$(python -m eval.eval_ppl \
        --hf_path $HF_CKPT \
        --overwrite_aquant $AQUANT \
        --manifest)
    
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation failed for ${bits}-bit"
        echo "$eval_output"
        return 1
    fi
    
    # Parse results and save to file
    echo "Parsing results..."
    echo "$eval_output" | grep -oP '<result>\K[^<]+' | while read -r result; do
        # Split by comma to get dataset and ppl
        dataset=$(echo "$result" | cut -d',' -f1 | xargs)
        ppl=$(echo "$result" | cut -d',' -f2 | xargs)
        echo "${bits},${dataset},${ppl}" >> $RESULTS_FILE
        echo "  ${dataset}: ${ppl}"
    done
    
    echo "Completed ${bits}-bit pipeline successfully!"
    echo ""
}

# Main execution
echo "Starting quantization and evaluation pipeline..."
echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Run for 2-bit
run_pipeline 2

# Run for 3-bit
run_pipeline 3

# Run for 4-bit  
run_pipeline 4

echo "All pipelines completed!"
echo "Results summary:"
cat $RESULTS_FILE 