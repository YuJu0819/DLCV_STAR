import torch
import json
from llama import ModelArgs, Tokenizer, Transformer
from pathlib import Path
from torchinfo import summary

def LLaMA_VQA(args, **kwargs):

    llama_weights = torch.load(args.llama_model_path, map_location='cpu')
    print(llama_weights.keys())
    model_args: ModelArgs = ModelArgs(max_seq_len=args.max_seq_len, max_batch_size=32, adapter_len=args.adapter_len, adapter_layer=args.adapter_layer)
    
    model_args.vocab_size = 32000
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model_llama_vqa = Transformer(model_args, args)
    torch.set_default_tensor_type(torch.FloatTensor)
    missing_keys, unexpected_keys =(model_llama_vqa.load_state_dict(llama_weights, strict=False))
    print('missing_keys:', missing_keys)
    print("unexpected_keys:", unexpected_keys)
    for name, param in model_llama_vqa.named_parameters():
        if ('gate1' in name) or ('gate2' in name) or ('adapter' in name) or ('temporal_emb' in name) or ('visual_proj' in name):
            param.requires_grad = True
            param.data = param.data.float()
        else:
            param.requires_grad = False
    
    summary(model_llama_vqa)

    return model_llama_vqa