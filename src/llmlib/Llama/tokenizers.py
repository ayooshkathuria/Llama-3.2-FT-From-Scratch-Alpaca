import os
from pathlib import Path

import tiktoken
from huggingface_hub import hf_hub_download
from tiktoken.load import load_tiktoken_bpe


class Llama3Tokenizer:
    def __init__(self, model_path):
        assert os.path.isfile(model_path), f"Model file {model_path} not found"
        mergeable_ranks = load_tiktoken_bpe(model_path)

        self.special_tokens = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.special_tokens.update(
            {
                f"<|reserved_{i}|>": 128002 + i
                for i in range(256)
                if (128002 + i) not in self.special_tokens.values()
            }
        )

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

    def encode(
        self,
        text,
        bos=False,
        eos=False,
        allowed_special=set(),
        disallowed_special=(),
    ):
        if bos:
            tokens = [self.special_tokens["<|begin_of_text|>"]]
        else:
            tokens = []

        tokens += self.model.encode(
            text,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )

        if eos:
            tokens.append(self.special_tokens["<|end_of_text|>"])
        return tokens

    def decode(self, tokens):
        return self.model.decode(tokens)


def load_llama3_tokenizer():
    """
    Load the tokenizer from the specified path.

    Parameters
    ----------
    tokenizer_path : str
        The path to the tokenizer file.

    Returns
    -------
    tokenizer : object
        The tokenizer object.
    """

    tokenizer_file_path = hf_hub_download(
        repo_id="meta-llama/Meta-Llama-3-8B",
        filename="original/tokenizer.model",
        local_dir="Llama-3.2-1B",
    )

    tokenizer = Llama3Tokenizer(tokenizer_file_path)

    return tokenizer
