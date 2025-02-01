import json
import urllib

import torch

model_configs = {
    "gpt2-small": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "norm": "layernorm",
        "hf_path": "openai-community/gpt2-small",
    },
    "gpt2-medium": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1024,
        "n_layers": 24,
        "n_heads": 16,
        "norm": "layernorm",
        "hf_path": "openai-community/gpt2-medium",
    },
    "gpt2-large": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1280,
        "n_layers": 36,
        "n_heads": 20,
        "norm": "layernorm",
        "hf_path": "openai-community/gpt2-large",
    },
    "gpt2-xl": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1600,
        "n_layers": 48,
        "n_heads": 25,
        "norm": "layernorm",
        "hf_path": "openai-community/gpt2-xl",
    },
    "llama2": {
        "vocab_size": 32000,  # Vocabulary size
        "context_length": 4096,  # Context length
        "emb_dim": 4096,  # Embedding dimension
        "n_heads": 32,  # Number of attention heads
        "n_layers": 32,  # Number of layers
        "hidden_dim": 11008,  # NEW: Size of the intermediate dimension in FeedForward
        "hf_path": "meta-llama/Llama-2-7b",
    },
    "llama_3_2_1B": {
        "vocab_size": 128_256,  # Vocabulary size
        "context_length": 512,  # Context length
        "emb_dim": 2048,  # NEW: Half the embedding dimension
        "n_heads": 32,  # Number of attention heads
        "n_layers": 16,  # NEW: Half the number of layers
        "hidden_dim": 8192,  # NEW: Almost half the size of the intermediate dimension in FeedForward
        "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
        "rope_base": 500000 // (15 * 16),  # The base in RoPE's "theta"
        "rope_freq": {  # RoPE frequency scaling
            "factor": 32.0,  # NEW: Adjustment of the rescaling factor
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 8192,
        },
        "tie_outhead_and_embedding": True,  # Tie the output head and the token embedding layer
        "hf_load_info": {
            "repo_id": "meta-llama/Llama-3.2-1B",
            "filename": "model.safetensors",
        },
    },
    "llama_3_2_3B": {
        "vocab_size": 128_256,  # Vocabulary size
        "context_length": 256,  # Context length
        "emb_dim": 3072,  # W: Half the embedding dimension
        "n_heads": 24,  # Number of attention heads
        "n_layers": 28,  # NEW: Half the number of layers
        "hidden_dim": 8192,  # NEW: Almost half the size of the intermediate dimension in FeedForward
        "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
        "rope_base": 500000 // (15 * 32),  # The base in RoPE's "theta"
        "rope_freq": {  # RoPE frequency scaling
            "factor": 32.0,  # NEW: Adjustment of the rescaling factor
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 8192,
        },
        "tie_outhead_and_embedding": True,  # Tie the output head and the token embedding layer
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
