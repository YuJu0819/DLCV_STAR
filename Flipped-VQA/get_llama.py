import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("princeton-nlp/Sheared-LLaMA-1.3B", low_cpu_mem_usage=True)
torch.save(model.state_dict(), 'llama.pth')
del model
