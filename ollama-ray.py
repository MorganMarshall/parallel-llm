import ray
import ollama

# Connect to the Ray cluster
ray.init(address="auto")

# Define the remote task that will generate text using Ollama
@ray.remote(num_gpus=1)  # Use a GPU if available
def generate_text(prompt):
    # Call Ollama's model to generate text based on the prompt
    response = ollama.chat(model="your-ollama-model", messages=[{"role": "user", "content": prompt}])
    return response['text']

# Parallel requests
prompts = [f"Prompt {i}" for i in range(20)]
futures = [generate_text.remote(prompt) for prompt in prompts]
results = ray.get(futures)

# Output the results
print(results)
