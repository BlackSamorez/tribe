AUTHOR=meta-llama
MODEL=Llama-3.2-3B

torchrun --nproc_per_node=8 --nnodes=1 -m quantize_llama.input_hessian_llama --base_model ${AUTHOR}/${MODEL} --save_path ~/hessians/${MODEL}-multihess --sample_proc 1 --batch_size 16