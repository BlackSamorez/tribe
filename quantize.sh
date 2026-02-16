python -m quantize_qwen.quantize_finetune_qwen \
  --base_model qwen/Qwen3-8B \
  --in_hess_path ~/hessians/Qwen3-8B \
  --save_path ~/models/QTIP/Qwen3-8B-4bit-3inst-gs128 \
  --group_size 128 \
  --L 16 --K 4 --V 1 --tlut_bits 0 --decode_mode 3inst \
  --ft_epochs 0