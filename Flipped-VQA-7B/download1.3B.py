import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("princeton-nlp/Sheared-LLaMA-1.3B")
print(model)
torch.save(model.state_dict(), "./pretrained/Sheared-LLaMA-1.3B.pth")
del model