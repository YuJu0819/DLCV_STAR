import torch
import json
from llama import ModelArgs, Tokenizer, Transformer
from pathlib import Path

def LLaMA_VQA(args, **kwargs):
    with open(f'{args.llama_model_path}/params.json', "r") as f:
        params = json.loads(f.read())
    print(f"Using model: {args.model}")
    
    checkpoints = (Path(args.llama_model_path)).glob("*.pth")
    
    loaded = []
    for x in checkpoints:
        print("loading from", x)
        loaded.append(torch.load(x, map_location="cuda"))
    
    if len(loaded) == 1:
        full_state_dict = loaded[0]

    model_args: ModelArgs = ModelArgs(max_seq_len=args.max_seq_len, max_batch_size=32, adapter_len=args.adapter_len, adapter_layer=args.adapter_layer, **params)
    
    model_args.vocab_size = 32000
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model_llama_vqa = Transformer(model_args, args)
    torch.set_default_tensor_type(torch.FloatTensor)
    missing_keys, unexpected_keys = model_llama_vqa.load_state_dict(full_state_dict, strict=False)

    for name, param in model_llama_vqa.named_parameters():
        if ('gate' in name) or ('adapter' in name) or ('temporal_emb' in name) or ('visual_proj' in name):
            param.requires_grad = True
            param.data = param.data.float()
        else:
            param.requires_grad = False

    return model_llama_vqa