import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Connect to the Ray cluster
ray.init(address="auto")

# Load LLM model on worker nodes
@ray.remote(num_gpus=1)
def generate_text(prompt):
    model_name = "your-small-llm"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=100)
    return tokenizer.decode(output[0])

# Parallel requests
prompts = [f"Prompt {i}" for i in range(20)]
futures = [generate_text.remote(prompt) for prompt in prompts]
results = ray.get(futures)

print(results)
