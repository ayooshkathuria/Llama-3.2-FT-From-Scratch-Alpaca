import json
import math
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint_sequential

from llmlib import GPT_ROOT
from llmlib.Llama.preloaded_llama import load_hf_weights_into_llama
from llmlib.Llama.tokenizers import Llama3Tokenizer
from llmlib.utils.models import model_configs
from llmlib.utils.prompts import format_w_alpaca


def precompute_rope_params(
    head_dim,
    theta_base=10_000,
    context_length=4096,
    freq_config=None,
    dtype=torch.float32,
):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        theta_base
        ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)] / head_dim).to(dtype)
    )

    ################################ NEW ###############################################
    # Frequency adjustments

    # We are basically going to divide our frequencies into 3 categories:
    # 1. Low Frequencies: These are basically the frequenies that correspond to higher dims and these help
    #    in capturing the global context as they don't repeat as frequently as the high frequency ones.
    #    This allows the network to have different values for positions that are far apart. We will basically
    #    divide these frequencies by a factor to further slow down how quickly values are repeated to enable
    #    the network to deal with longer contexts. Generally, if the original context was x and we want to
    #    make the network work for a context of length y, we would divide the frequencies by x/2y.
    #
    # 2. High Frequencies: These are the frequencies that correspond to lower dims and these help in capturing
    #    the local context as they repeat more frequently. Since these frequencies change more rapidly, they
    #    help the network differentiate between variations across positions that are close to each other. We
    #    keep this frequency as is.
    #
    # 3. Medium Frequencies: These are the frequencies that correspond to the dims that are in between the low and
    #    high frequencies. We will interpolate the scaling factor between the low and high frequencies to get a
    #   smooth transition between the two.

    if freq_config is not None:
        # Get the low end of the wavelength. Frequencies that are smaller than these will be further divided to
        # to make them low enough for the larger context.
        low_freq_wavelen = (
            freq_config["original_context_length"] / freq_config["low_freq_factor"]
        )

        # Now compute the high end of the wavelength. Frequencies that are larger than these will be kept as is.
        high_freq_wavelen = (
            freq_config["original_context_length"] / freq_config["high_freq_factor"]
        )

        # COmpute the wavelength corresponding to all the frequencies.
        wavelen = 2 * torch.pi / inv_freq

        # Now we will adjust the frequencies based on the wavelength. If the wavelength is smaller than the low end
        # of the wavelength, we will divide the frequency by the low frequency factor.
        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        # Now we will do the smoothening between two curves. The curve behaves.
        # 1. Freq / factor for wave > low_freq_wavelen or freq < 2 * pi / low_freq_wavelen
        # 2. Freq for wave < low_freq_wavelen or freq > 2 * pi / low_freq_wavelen.
        # We will carry out the smoothening between freq correspond to low_freq_wavelen and high_freq_wavelen.

        # To understand, notice that the term (freq_config["original_context_length"] / wavelen) ranges from
        # freq_config["high_freq_factor"] to freq_config["low_freq_factor"] as wavelen is increased from low
        # to high. This means smoothening factor will range from 0 to 1 as wavelen is increased from low to high.
        smooth_factor = (
            freq_config["original_context_length"] / wavelen
            - freq_config["low_freq_factor"]
        ) / (freq_config["high_freq_factor"] - freq_config["low_freq_factor"])

        # We use polyak averaging to smoothen the transition between the two curves.
        # At the high frequency end, smoothing factor is 1, and prefers to use the original frequency.
        # At the low frequency end, smoothing factor is 0, and prefers to use scaled frequency.

        smoothed_inv_freq = (1 - smooth_factor) * (
            inv_freq / freq_config["factor"]
        ) + smooth_factor * inv_freq

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama
    ####################################################################################

    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    angles = (
        positions[:, None] * inv_freq[None, :]
    )  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    _, _, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated


class SharedBuffers:
    _kv_buffers = {}  # a class variable instead of an instance variable.

    @staticmethod
    def get_buffers(
        context_length, head_dim, rope_base, freq_config, dtype=torch.float32
    ):
        key = (
            context_length,
            head_dim,
            rope_base,
            tuple(freq_config.values()) if freq_config else freq_config,
        )

        if key not in SharedBuffers._kv_buffers:
            # Create or fetch the buffers
            mask = torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1,
            )
            cos, sin = precompute_rope_params(
                head_dim, rope_base, context_length, freq_config, dtype
            )
            SharedBuffers._kv_buffers[key] = (mask, cos, sin)

        return SharedBuffers._kv_buffers[key]


class CausalAttention(nn.Module):
    def __init__(
        self, d_in, d_out, context_length, n_heads, n_kv_groups, rope_base, rope_config
    ):
        super(CausalAttention, self).__init__()

        assert d_out % n_heads == 0, "d_out must be divisible by d_heads"
        assert n_heads % n_kv_groups == 0, "n_heads must be divisible by n_kv_groups"

        self.d_out = d_out
        self._n_heads = n_heads
        self._head_dim = d_out // n_heads
        self._n_kv_groups = n_kv_groups
        self._group_size = n_heads // n_kv_groups
        self._context_length = context_length

        # self.c_attn = nn.Linear(d_in, 3 * d_out, bias=bias)
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key = nn.Linear(d_in, self._n_kv_groups * self._head_dim, bias=False)
        self.W_value = nn.Linear(d_in, self._n_kv_groups * self._head_dim, bias=False)

        self.out_projection = nn.Linear(d_out, d_out, bias=False)
        # self.register_buffer(
        #     "mask",
        #     1 - torch.tril(torch.ones(context_length, context_length)),
        # )

        # Compute RoPE (Relative positional encoding) cos and sin buffers.
        mask, cos, sin = SharedBuffers.get_buffers(
            context_length,
            self._head_dim,
            rope_base,
            rope_config,
            dtype=self.W_query.weight.dtype,
        )

        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        self._enable_kv_caching = False

        # self.k_cache = torch.zeros(
        #     1, self._n_kv_groups, self._context_length, self._head_dim
        # ).to(self.W_query.weight.dtype)

        # self.v_cache = torch.zeros(
        #     1, self._n_kv_groups, self._context_length, self._head_dim
        # ).to(self.W_query.weight.dtype)

    def enable_kv_caching(self):

        dtype = self.W_query.weight.dtype
        device = self.W_query.weight.device

        self.k_cache = torch.zeros(
            1, self._n_kv_groups, self._context_length, self._head_dim
        ).to(device=device, dtype=dtype)

        self.v_cache = torch.zeros(
            1, self._n_kv_groups, self._context_length, self._head_dim
        ).to(device=device, dtype=dtype)

        self._enable_kv_caching = True

    def disable_kv_caching(self):
        # Warn the user this will delete the kv cache and enabling it again would involve
        # rellocation for the cache.
        self._enable_kv_caching = False

        del self.k_cache
        del self.v_cache

    def is_kv_caching_enabled(self):
        return self._enable_kv_caching

    def forward(self, x, start_pos):
        # qkv = self.c_attn(x)
        # q, k, v = torch.chunk(qkv, 3, dim=-1)

        b, t, e = x.shape

        q, k, v = self.W_query(x), self.W_key(x), self.W_value(x)

        # TO:DO
        # return (
        # x[:, :, :, None, :]
        # .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        # .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        # )

        q = q.view(b, t, self._n_heads, self._head_dim)
        k = k.view(b, t, self._n_kv_groups, self._head_dim)
        v = v.view(b, t, self._n_kv_groups, self._head_dim)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if self._enable_kv_caching:
            # Check if start_pos + t is greater than the context length.
            if start_pos + t > self._context_length:
                start_pos = self._context_length - t
                self.k_cache = torch.roll(self.k_cache, -t, dims=2)
                self.v_cache = torch.roll(self.v_cache, -t, dims=2)

        k = compute_rope(
            k, self.cos[start_pos : start_pos + t], self.sin[start_pos : start_pos + t]
        )
        q = compute_rope(
            q, self.cos[start_pos : start_pos + t], self.sin[start_pos : start_pos + t]
        )

        n_rep = self._n_heads // self._n_kv_groups

        if self._enable_kv_caching:
            self.k_cache[:, :, start_pos : start_pos + t] = k
            self.v_cache[:, :, start_pos : start_pos + t] = v

            k = self.k_cache[:, :, : start_pos + t]
            v = self.v_cache[:, :, : start_pos + t]

        # Duplicate the key and value tensors to match the number of heads.
        # k = k.repeat_interleave(
        #     self._group_size, dim=1
        # )  # we interleave because of how the weights are.
        # v = v.repeat_interleave(self._group_size, dim=1)

        k = (
            k[:, :, None, :, :]
            .expand(b, self._n_kv_groups, n_rep, start_pos + t, self._head_dim)
            .reshape(b, n_rep * self._n_kv_groups, start_pos + t, self._head_dim)
        )

        v = (
            v[:, :, None, :, :]
            .expand(b, self._n_kv_groups, n_rep, start_pos + t, self._head_dim)
            .reshape(b, n_rep * self._n_kv_groups, start_pos + t, self._head_dim)
        )

        # Compute the attention weights.
        attention_scores = q @ k.transpose(-2, -1)

        if t > 1:
            attention_scores = torch.masked_fill(
                attention_scores, self.mask.bool()[:t, :t], -torch.inf
            )

        attention_scores = attention_scores / (k.shape[-1] ** 0.5)

        # Normalise them to get attention scores.
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply the attention weights to the values.
        context_vec = attention_weights @ v

        context_vec = context_vec.transpose(1, 2).contiguous().view(b, t, self.d_out)

        # Project the output back to the original dimension.
        context_vec = self.out_projection(context_vec)

        return context_vec


class FeedForwardwSwiGLU(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super(TransformerBlock, self).__init__()

        # Layer norm layers.
        self.ln1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)
        self.ln2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)

        # self.ln1 = RMSNorm(cfg["emb_dim"], eps=1e-5)
        # self.ln2 = RMSNorm(cfg["emb_dim"], eps=1e-5)

        self.attn = CausalAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            n_heads=cfg["n_heads"],
            n_kv_groups=cfg["n_kv_groups"],
            rope_base=cfg["rope_base"],
            rope_config=cfg["rope_freq"],
        )

        self.ff = FeedForwardwSwiGLU(cfg["emb_dim"], cfg["hidden_dim"])

    def forward(self, x, start_pos=0):

        x = x + self.attn(self.ln1(x), start_pos)
        x = x + self.ff(self.ln2(x))

        return x


class Llama3(nn.Module):
    def __init__(self, cfg):
        super(Llama3, self).__init__()

        self.cfg = cfg

        self.token_embedding_layer = torch.nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"]
        )

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5)
        # self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-5)

        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        if cfg.get("tie_outhead_and_embedding", False):
            self.out_head.weight = self.token_embedding_layer.weight

            assert torch.allclose(
                self.out_head.weight, self.token_embedding_layer.weight
            )

        self.loss = nn.CrossEntropyLoss()

        self.enable_gradient_checkpoints = cfg.get(
            "enable_gradient_checkpointing", False
        )

        self._kv_caching_enabled = False

    def forward(self, inp_tk_ids, start_pos=0, targets=None):
        x = self.token_embedding_layer(inp_tk_ids)

        if self.enable_gradient_checkpoints:
            x = checkpoint_sequential(
                self.transformer_blocks,
                self.cfg["n_layers"],
                x,
                use_reentrant=False,
            )
        else:
            for transformer_block in self.transformer_blocks:
                x = transformer_block(x, start_pos)

        x = self.final_norm(x)

        logits = self.out_head(x)

        if targets is not None:
            loss = self.loss(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            return logits

    def enable_kv_caching(
        self,
    ):
        # Loop over all the

        for module in self.modules():
            # If it's a causal attention module, enable kv caching.
            if isinstance(module, CausalAttention):
                module.enable_kv_caching()

        self._kv_caching_enabled = True

    def disable_kv_caching(self):
        # Loop over all the

        for module in self.modules():
            # If it's a causal attention module, enable kv caching.
            if isinstance(module, CausalAttention):
                module.disable_kv_caching()

        self._kv_caching_enabled = False

    def generate(
        self,
        inp_tk_ids,
        max_new_token,
        context_length,
        top_k=None,
        temperature=0.0,
        eos_id=None,
        top_p=None,
        repetition_penalty=1.0,
    ):

        inp_tk_ids = torch.atleast_2d(inp_tk_ids).long()

        inp_tk_ids = inp_tk_ids[:, -context_length:]

        start = 0

        end = inp_tk_ids.shape[1]

        for i in range(max_new_token):

            with torch.no_grad():
                # if i == 0:
                #     logits = self(inp_tk_ids)
                # else:
                #     logits = self(inp_tk_ids[:, -1:])
                logits = self(inp_tk_ids[:, start:end], start)

            logits = logits[:, -1, :]

            # return logits

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(logits.size(0)):
                    for previous_token in set(inp_tk_ids[i].tolist()):
                        logits[i, previous_token] /= repetition_penalty

            if top_k is not None:
                top_logits = torch.topk(logits, top_k, dim=-1)
                min_val = top_logits.values[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float("-inf")).to(logits.device),
                    logits,
                )

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(
                    torch.softmax(
                        sorted_logits / (temperature if temperature > 0.0 else 1),
                        dim=-1,
                    ),
                    dim=-1,
                )

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()

                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            if temperature > 0.0:
                logits = logits / temperature

                probs = torch.softmax(logits, dim=-1)

                next_inp_tk_id = torch.multinomial(probs, num_samples=1)
            else:
                next_inp_tk_id = torch.argmax(logits, dim=-1, keepdim=True)

            if self._kv_caching_enabled:
                start = end
            else:
                if end > context_length:
                    start += 1

            end += 1

            if eos_id is not None and next_inp_tk_id.item() in eos_id:
                break

            inp_tk_ids = torch.cat([inp_tk_ids, next_inp_tk_id], dim=-1)

        return inp_tk_ids


if __name__ == "__main__":
    # batch_size = 1
    # context_len = 100
    # max_context_len = 4096
    # embed_dim = 128
    # num_heads = 4

    # example_batch = torch.randn((batch_size, context_len, embed_dim))

    # mha = CausalAttention(
    #     d_in=embed_dim,
    #     d_out=embed_dim,
    #     context_length=max_context_len,
    #     n_heads=num_heads,
    # )

    # a = mha(example_batch)

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")

    torch.manual_seed(999)

    model = Llama3(model_configs["llama_3_2_3B"])

    # Get the number of elements in the model
    num_params = sum(p.numel() for p in model.parameters())

    print(num_params)

    from huggingface_hub import hf_hub_download

    tokenizer_file_path = hf_hub_download(
        repo_id="meta-llama/Meta-Llama-3-8B",
        filename="original/tokenizer.model",
        local_dir="Llama-3.2-1B",
    )

    tokenizer = Llama3Tokenizer(tokenizer_file_path)

    combined_weights = {}

    for i in range(1, 3):
        weights_file = hf_hub_download(
            repo_id="meta-llama/Llama-3.2-3B",
            filename=f"model-0000{i}-of-00002.safetensors",
            local_dir="Llama-3.2-3B",
        )
        current_weights = load_file(weights_file)
        combined_weights.update(current_weights)

    weights = combined_weights
    load_hf_weights_into_llama(model, model_configs["llama_3_2_3B"], weights)

    model = model.to(torch.get_default_device())

    torch.manual_seed(999)

    prompt = "Every effort"

    # Load the alpaca dataset
    # data = json.load(open("data/alpaca_data_cleaned.json"))

    # prompt = format_w_alpaca(data[645])

    token_ids = torch.tensor([tokenizer.encode(prompt)]).long().to("cuda")

    a = time.time()

    # model.enable_kv_caching()

    # token_ids = model.generate(
    #     token_ids,
    #     max_new_token=25,
    #     context_length=model_configs["llama_3_2_1B"]["context_length"],
    #     top_k=1,
    #     temperature=0.0,
    #     eos_id=[128001],
    # )

    # b = time.time()

    # print(b - a)

    # print("Output text:\n", tokenizer.decode(token_ids[0].tolist()))


# Write the code for a LORALayer
class LORALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super(LORALayer, self).__init__()

        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLORA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super(LinearWithLORA, self).__init__()

        self.linear = linear
        self.lora = LORALayer(
            self.linear.in_features, self.linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


import torch


def merge_lora_weights(linear_with_lora: LinearWithLORA):
    """
    Merges the LoRA weights into the original linear layer weights.

    Args:
        linear_with_lora (LinearWithLORA): An instance of LinearWithLORA where LoRA weights are to be merged.

    Returns:
        torch.nn.Linear: The original linear layer with LoRA weights merged.
    """
    # Extract components
    linear = linear_with_lora.linear
    lora_layer = linear_with_lora.lora

    # Calculate the LoRA contribution
    lora_contribution = (lora_layer.alpha * (lora_layer.A @ lora_layer.B)).T

    # Merge the LoRA weights into the linear weights
    linear.weight.data += lora_contribution

    # Return the merged linear layer
    return linear


def replace_lora_with_linear(model):
    for name, module in model.named_children():
        if isinstance(module, LinearWithLORA):
            setattr(model, name, merge_lora_weights(module))
        else:
            replace_lora_with_linear(module)


def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearWithLORA(module, rank, alpha).to("cuda"))
        else:
            replace_linear_with_lora(module, rank, alpha)


def create_llama3_model(config):

    model_cfg = model_configs[config["foundation_model"]]

    model = Llama3(model_cfg)

    # weights_file = hf_hub_download(
    #     repo_id=model_cfg["hf_load_info"]["repo_id"],
    #     filename=model_cfg["hf_load_info"]["filename"],
    #     local_dir=config["foundation_model"],
    # )

    # weights = load_file(weights_file)

    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["hf_load_info"]["repo_id"]
    )

    weights = hf_model.state_dict()

    load_hf_weights_into_llama(model, model_cfg, weights)

    del weights

    if config.get("preload_model", False):
        with open(GPT_ROOT / config["model_save_path"], "rb") as f:
            model.load_state_dict(torch.load(f))

    if config.get("enable_lora", False):
        # Freeze the parameters of the model.
        for param in model.parameters():
            param.requires_grad = False

        # Replace the linear layers with LinearWithLora layers.
        replace_linear_with_lora(model, 8, 16)

        num_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        print(f"Number of trainable parameters: {num_trainable_params}")

    model = model.to(config["device"]).eval()

    return model
