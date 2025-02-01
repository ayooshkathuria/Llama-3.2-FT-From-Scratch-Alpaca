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
    project_name: str
    experiment_name: str
    data_path: str
    # vocab_size: int
    # context_length: int
    max_seq_len: int
    drop_rate: float
    qkv_bias: bool
    device: str
    foundation_model: str
    tokenizer: str
    seed: int
    lr: float
    lr_scheduling: dict
    batch_size: int
    weight_decay: float = 0.01
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
    use_deepspeed: bool = False
    use_8bit_optim: bool = False

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
