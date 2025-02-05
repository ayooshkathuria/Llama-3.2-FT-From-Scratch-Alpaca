import json
import urllib

import torch

model_configs = {
    "llama_3_2_1B": {
        "vocab_size": 128_256,
        "context_length": 512,
        "emb_dim": 2048,
        "n_heads": 32,
        "n_layers": 16,
        "hidden_dim": 8192,
        "n_kv_groups": 8,
        "rope_base": 500000 // (15 * 16),
        "rope_freq": {
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 8192,
        },
        "tie_outhead_and_embedding": True,
        "hf_load_info": {
            "repo_id": "meta-llama/Llama-3.2-1B",
            "filename": "model.safetensors",
        },
    },
    "llama_3_2_3B": {
        "vocab_size": 128_256,
        "context_length": 256,
        "emb_dim": 3072,
        "n_heads": 24,
        "n_layers": 28,
        "hidden_dim": 8192,
        "n_kv_groups": 8,
        "rope_base": 500000 // (15 * 32),
        "rope_freq": {
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 8192,
        },
        "tie_outhead_and_embedding": True,
        "hf_load_info": {
            "repo_id": "meta-llama/Llama-3.2-3B",
            "filename": "model.safetensors",
        },
    },
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def query_ollama_model(prompt, model="llama3.1", url="http://localhost:11434/api/chat"):
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {  # Settings below are required for deterministic responses
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048,
        },
    }

    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data
