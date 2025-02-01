import torch.nn as nn


def check_and_copy(target, source):
    device = target.device
    dtype = target.dtype
    if target.shape == source.shape:
        target = nn.Parameter(source.clone().detach().to(device=device, dtype=dtype))
        return target
    else:
        raise ValueError(f"Shapes do not match: {target.shape} != {source.shape}")


def load_hf_weights_into_llama(model, param_config, params):
    model.token_embedding_layer.weight = check_and_copy(
        model.token_embedding_layer.weight, params["model.embed_tokens.weight"]
    )

    for l in range(param_config["n_layers"]):
        # Load attention weights
        model.transformer_blocks[l].attn.W_query.weight = check_and_copy(
            model.transformer_blocks[l].attn.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
        )

        model.transformer_blocks[l].attn.W_key.weight = check_and_copy(
            model.transformer_blocks[l].attn.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
        )
        model.transformer_blocks[l].attn.W_value.weight = check_and_copy(
            model.transformer_blocks[l].attn.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
        )
        model.transformer_blocks[l].attn.out_projection.weight = check_and_copy(
            model.transformer_blocks[l].attn.out_projection.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
        )
        model.transformer_blocks[l].ln1.weight = check_and_copy(
            model.transformer_blocks[l].ln1.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
        )

        # Load FeedForward weights
        model.transformer_blocks[l].ff.fc1.weight = check_and_copy(
            model.transformer_blocks[l].ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
        )

        # For some reason w2 and w3 are provided in the wrong order in the weights file
        model.transformer_blocks[l].ff.fc2.weight = check_and_copy(
            model.transformer_blocks[l].ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
        )
        model.transformer_blocks[l].ff.fc3.weight = check_and_copy(
            model.transformer_blocks[l].ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
        )
        model.transformer_blocks[l].ln2.weight = check_and_copy(
            model.transformer_blocks[l].ln2.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
        )

    # Load output layer weights
    model.final_norm.weight = check_and_copy(
        model.final_norm.weight, params["model.norm.weight"]
    )

    if "lm_head.weight" in params.keys():
        model.out_head.weight = check_and_copy(
            model.out_head.weight, params["lm_head.weight"]
        )
    else:
        model.out_head.weight = check_and_copy(
            model.out_head.weight,
            params["model.embed_tokens.weight"],
        )


if __name__ == "__main__":
    pass
