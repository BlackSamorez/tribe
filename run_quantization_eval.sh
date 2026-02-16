#!/bin/bash

# Create results file
DECODE_MODE="3inst_fp8"
AQUANT="fp8"
GROUP_SIZE="128"
RESULTS_FILE="results_${DECODE_MODE}_${AQUANT}_gs${GROUP_SIZE}.txt"
echo "Bit,Dataset,PPL" > $RESULTS_FILE

# Function to run full pipeline for a given bit width
run_pipeline() {
    local bits=$1
    echo "Running pipeline for ${bits}-bit quantization..."
    
    # Step 1: Quantize
    echo "Step 1: Quantizing model to ${bits}-bit..."
    python -m quantize_llama.quantize_finetune_llama \
        --base_model meta-llama/Llama-3.1-8B \
        --in_hess_path ~/hessians/Llama-3.1-8B \
        --save_path ~/models/QTIP/Llama-3.1-8B-${bits}bit-${DECODE_MODE}-${AQUANT}-gs${GROUP_SIZE} \
        --group_size $GROUP_SIZE \
        --aquant $AQUANT \
        --L 16 --K $bits --V 1 --tlut_bits 0 --decode_mode $DECODE_MODE \
        --ft_epochs 0
    
    if [ $? -ne 0 ]; then
        echo "Error: Quantization failed for ${bits}-bit"
        return 1
    fi
    
    # Step 2: Convert to HF format
    echo "Step 2: Converting to HuggingFace format..."
    python -m quantize_llama.hfize_llama \
        --quantized_path ~/models/QTIP/Llama-3.1-8B-${bits}bit-${DECODE_MODE}-${AQUANT}-gs${GROUP_SIZE} \
        --hf_output_path ~/models/QTIP/Llama-3.1-8B-${bits}bit-${DECODE_MODE}-${AQUANT}-gs${GROUP_SIZE}-hf
    
    if [ $? -ne 0 ]; then
        echo "Error: HF conversion failed for ${bits}-bit"
        return 1
    fi
    
    # Step 3: Evaluate and capture results
    echo "Step 3: Evaluating perplexity..."
    eval_output=$(python -m eval.eval_ppl \
        --hf_path ~/models/QTIP/Llama-3.1-8B-${bits}bit-${DECODE_MODE}-${AQUANT}-gs${GROUP_SIZE}-hf \
        --manifest 2>&1)
    
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