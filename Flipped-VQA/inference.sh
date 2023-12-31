python3 inference.py \
    --batch_size 1 \
    --llama_model_path llama.pth \
    --adapter_layer 24 \
    --max_seq_len 128 \
    --dataset star \
    --output_dir output_dir \
    --resume model.pth
