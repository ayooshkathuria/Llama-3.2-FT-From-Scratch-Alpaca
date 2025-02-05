import random
import string
from dataclasses import dataclass

from llmlib.utils.models import get_device, model_configs


def flatten_dict(d, parent_key="", sep="/"):
    """
    Flatten a nested dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key string for recursion.
        sep (str): The separator to use between keys.

    Returns:
        dict: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def generate_random_string(length=6):
    """Generates a random string of specified length."""
    return "".join(random.choices(string.digits, k=length))


@dataclass
class FineTuningConfig:
    """
    Parameters
    ----------
    project_name : str
        Name of the project.
    experiment_name : str
        Name of the experiment. A random string will be appended to this.
    data_path : str
        Path to the training data.
    max_seq_len : int
        Maximum sequence length for input tokens.
    drop_rate : float
        Dropout rate to use in the model.
    qkv_bias : bool
        Whether to use bias in QKV attention calculations.
    device : str
        Device to run training on ('cpu', 'cuda', etc.).
    foundation_model : str
        Name/type of the base model to fine-tune. The list of available models can be found in the
        `model_configs` dictionary. from the `llmlib.utils.models` module.
    seed : int
        Random seed for reproducibility.
    lr : float
        Learning rate for training.
    lr_scheduling : dict
        Learning rate scheduling configuration.
    batch_size : int
        Training batch size.
    max_gen_tokens : int, optional
        Maximum number of tokens to generate during inference. Defaults to 256.
    weight_decay : float, optional
        Weight decay coefficient. Defaults to 0.01.
    inference_only : bool, optional
        Whether to run in inference-only mode. Defaults to False.
    preload_model : bool, optional
        Whether to preload the model. Defaults to False.
    enable_lora : bool, optional
        Whether to enable LoRA fine-tuning. Defaults to False.
    num_epochs : int or None, optional
        Number of training epochs. Defaults to None.
    num_train_iters : int or None, optional
        Number of training iterations. Defaults to None.
    eval_batch_size : int or None, optional
        Batch size for evaluation. Defaults to training batch_size.
    gradient_accumulation_steps : int, optional
        Number of gradient accumulation steps. Defaults to 1.
    eval_freq : int or None, optional
        Frequency of evaluation in iterations. Defaults to None.
    print_memory_usage : bool, optional
        Whether to print memory usage statistics. Defaults to False.
    generation_freq : int or None, optional
        Frequency of text generation in iterations. Defaults to None.
    responses_save_path : str or None, optional
        Path to save generated responses. Defaults to None.
    model_save_path : str or None, optional
        Path to save model checkpoints. Defaults to None.
    log_to_clearml : bool, optional
        Whether to log metrics to ClearML. Defaults to False.
    enable_gradient_checkpointing : bool, optional
        Whether to enable gradient checkpointing. Defaults to False.
    use_bf16 : bool, optional
        Whether to use bfloat16 precision. Defaults to False.
    use_explicit_bfloat16 : bool, optional
        Whether to explicitly use bfloat16 dtype. Defaults to False.
    use_8bit_optim : bool, optional
        Whether to use 8-bit optimization. Defaults to False.
    use_ollama_for_eval : bool, optional
        Whether to use OLLAMA for evaluation. Defaults to False.
    ollama_model_name : str, optional
        Name of the OLLAMA model to use. Defaults to 'llama3.1'.

    Notes
    -----

    - Either num_train_iters or num_epochs must be provided, but not both.
     - If lr_scheduling is provided, init_lr and eta_min will default to lr if not specified.
     - Model-specific configurations from model_configs will be added to the instance attributes.

    """

    project_name: str
    experiment_name: str
    data_path: str
    max_seq_len: int
    drop_rate: float
    qkv_bias: bool
    device: str
    foundation_model: str
    seed: int
    lr: float
    lr_scheduling: dict
    batch_size: int
    max_gen_tokens: int = 256
    weight_decay: float = 0.01
    inference_only: bool = False
    preload_model: bool = False
    enable_lora: bool = False
    num_epochs: int | None = None
    num_train_iters: int | None = None
    eval_batch_size: int | None = None
    gradient_accumulation_steps: int = 1
    eval_freq: int | None = None
    print_memory_usage: bool = False
    generation_freq: int | None = None
    responses_save_path: str | None = None
    model_save_path: str | None = None
    log_to_clearml: bool = False
    enable_gradient_checkpointing: bool = False
    use_bf16: bool = False
    use_explicit_bfloat16: bool = False
    use_8bit_optim: bool = False
    use_ollama_for_eval: bool = False
    ollama_model_name: str = "llama3.1"

    def __post_init__(self):
        # Generate a random string of 6 characters and suffix it to
        # the experiment name.
        self.experiment_name = f"{self.experiment_name}-{generate_random_string()}"

        # Both num_train_iter and num_epochs cannot be provided.
        assert self.num_train_iters is not None or self.num_epochs is not None

        # If neither is provided, raise an error.
        if self.num_train_iters is None and self.num_epochs is None:
            raise AssertionError(
                "Either num_train_iter or num_epochs must be provided."
            )

        # If lr_scheduling is provided, make sure that init_lr and eta_min are present.
        # If not, set them to the default lr value.
        if self.lr_scheduling:
            self.lr_scheduling["init_lr"] = self.lr_scheduling.get("init_lr", self.lr)
            self.lr_scheduling["eta_min"] = self.lr_scheduling.get("eta_min", self.lr)

        # If eval_batch_size is not provided, set it to the training batch size.
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size

        # We add the model config of the foundation model to be used to the config.
        # self.__dict__.update(model_configs[self.foundation_model])

        for key, value in model_configs[self.foundation_model].items():
            setattr(self, key, value)

        print
