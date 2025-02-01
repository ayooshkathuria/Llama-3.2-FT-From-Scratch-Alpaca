import copy
import functools
import gc
import json

# Set the envrironment variablee PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# to allow for dynamic memory allocation on the GPU.
import os
import random
import string
import time
import urllib
from dataclasses import dataclass, field

import bitsandbytes as bnb
import psutil
import tiktoken
import torch
from clearml import StorageManager, Task
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

# torch.set_float32_matmul_precision("high")
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm, trange
from transformers import GPT2Model

from llmlib import GPT_ROOT
from llmlib.Llama.llama import Llama3
from llmlib.Llama.preloaded_llama import load_hf_weights_into_llama
from llmlib.Llama.tokenizers import Llama3Tokenizer
from llmlib.utils.eval import (
    evaluate_w_ollama,
    evaluate_with_promestheus,
    generate_response,
)
from llmlib.utils.lr_schedulers import WarmupCosineAnnealingLR
from llmlib.utils.models import get_device, model_configs
from llmlib.utils.prompts import format_w_alpaca
from llmlib.utils.utils import FineTuningConfig, flatten_dict

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class InstructionDataset:
    """
    A dataset class for handling instruction data and encoding it using a tokenizer.

    Parameters
    ----------
    data : list
        A list of data entries to be processed and encoded.
    tokenizer : object
        A tokenizer object with an `encode` method to convert text into token IDs.

    Attributes
    ----------
    data : list
        The original list of data entries.
    encoded_texts : list
        A list of encoded texts, where each text is represented as a list of token IDs..

    Methods
    -------
    __getitem__(idx)
        Returns the encoded text at the specified index.
    __len__()
        Returns the number of encoded texts in the dataset.
    """

    def __init__(self, data, tokenizer):
        """Initializes the InstructionDataset object."""
        self.data = data
        self.encoded_texts = []

        for entry in data:
            full_text = format_w_alpaca(entry, get_response=True)

            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, idx):
        return self.encoded_texts[idx]

    def __len__(self):
        return len(self.encoded_texts)


def custom_collate_fn(
    batch, pad_token_id=128001, ignore_index=-100, device="cpu", allowed_max_length=1024
):
    """
    Custom collate function to prepare batches for training.

    Parameters
    ----------
    batch : list of list of int
        A batch of sequences, where each sequence is a list of token IDs.
    pad_token_id : int, optional
        The token ID used for padding sequences (default is 50256).
    ignore_index : int, optional
        The index that will be ignored in the loss computation (default is -100).
    device : str, optional
        The device to which the tensors will be moved (default is "cpu").
    allowed_max_length : int, optional
        The maximum allowed length for the sequences. If provided, sequences will be truncated to this length (default is None).

    Returns
    -------
    inputs_tensor : torch.Tensor
        A tensor of shape (batch_size, sequence_length) containing the input sequences, padded and truncated as necessary.
    targets_tensor : torch.Tensor
        A tensor of shape (batch_size, sequence_length) containing the target sequences, padded, truncated, and with ignore_index applied as necessary.
    """

    batch_max_length = max([len(item) + 1 for item in batch])

    inputs_lst = []
    targets_lst = []

    for item in batch:
        new_item = item.copy()

        new_item += [pad_token_id]

        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        ignore_indexes_from = len(item)
        if ignore_indexes_from < batch_max_length:
            targets[ignore_indexes_from:] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def compute_loss(model, dataloader, device="cpu", num_batches=None):
    model.eval()

    total_loss = 0

    if num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))

    batches_processed = 0

    for input, target in tqdm(dataloader, leave=False, dynamic_ncols=True):
        input, target = input.to(device), target.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(input, targets=target)

        total_loss += loss.item()

        batches_processed += 1

        if batches_processed >= num_batches:
            break

    avg_loss = total_loss / batches_processed

    return avg_loss


def configure_optimizer(
    model, lr, weight_decay_2d=0.1, weight_decay_1d=0.0, use_8bit_optim=False
):
    """
    Configures the optimizer with different weight decay values for parameters
    based on their dimensions.

    Parameters
    ----------
    model : torch.nn.Module
        The model containing the parameters to optimize.
    lr : float
        The learning rate for the optimizer.
    weight_decay_2d : float, optional
        Weight decay for parameters with more than 2 dimensions (default is 0.1).
    weight_decay_1d : float, optional
        Weight decay for 1-dimensional parameters (default is 0.0).

    Returns
    -------
    optimizer : torch.optim.Optimizer
        The configured optimizer.
    """
    params_with_decay = []
    params_without_decay = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.ndimension() > 1:
                params_with_decay.append(param)
            else:
                params_without_decay.append(param)

    if use_8bit_optim:
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=lr, weight_decay=weight_decay_2d
        )
    else:

        optimizer = torch.optim.AdamW(
            [
                {"params": params_with_decay, "weight_decay": weight_decay_2d},
                {"params": params_without_decay, "weight_decay": weight_decay_1d},
            ],
            lr=lr,
            fused=True,
        )

    return optimizer


def llama_trainer(
    config: dict,
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    tokenizer: tiktoken.core.Encoding,
    test_data,
    num_train_iters: int | None = None,
    num_epochs: int | None = None,
    eval_freq: int = 1,
    generation_freq: int | None = 1,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Trains a GPT model using the provided dataloaders and optimizer.

    Parameters
    ----------
    model : torch.ngpt_modeln.Module
        The GPT model to be trained.
    train_dataloader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    val_dataloader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    optimizer : torch.optim.Optimizer
        Optimizer for updating the model parameters.
    num_train_iter : int, optional
        Number of training iterations. Either `num_train_iter` or `num_epochs` must be provided.
    num_epochs : int, optional
        Number of epochs to train the model. Either `num_train_iter` or `num_epochs` must be provided.
    eval_freq : int, default=1
        Frequency (in steps) at which to evaluate the model on the validation dataset.
    generation_freq : int or None, default=1
        Frequency (in steps) at which to generate responses using the model. If None, no responses are generated.
    device : str, default="cpu"
        Device on which to perform training (e.g., "cpu" or "cuda").

    Returns
    -------
    model : torch.nn.Module
        The trained GPT model.

    Raises
    ------
    AssertionError
        If neither `num_train_iter` nor `num_epochs` is provided.

    Notes
    -----
    The function performs training in a loop, evaluating the model and generating responses at specified intervals.
    """

    global_step = 0
    epoch = 1
    trainloader_iterator = iter(train_dataloader)

    if num_epochs is not None:
        num_train_iters = len(train_dataloader) * num_epochs

    running_train_loss = 0
    tokens_processed = 0

    max_seq_len = -1

    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

    scheduler = None

    optimizer = configure_optimizer(
        model,
        lr=config["lr"],
        weight_decay_2d=config["weight_decay"],
        weight_decay_1d=0.0,
        use_8bit_optim=config["use_8bit_optim"],
    )

    # Init the scheduler.
    if "lr_scheduling" in config.keys():
        warmup_steps = int(
            config["lr_scheduling"]["warmup_percentage"] * num_train_iters
        )

        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=num_train_iters,
            init_lr=config["lr_scheduling"]["init_lr"],
            eta_min=config["lr_scheduling"]["eta_min"],
        )

    optimizer.zero_grad()

    learning_rates = []

    t0 = time.time()

    for step in (pbar := trange(num_train_iters, disable=False, dynamic_ncols=True)):
        model.train()

        try:
            input, target = next(trainloader_iterator)
        except StopIteration:
            trainloader_iterator = iter(train_dataloader)
            input, target = next(trainloader_iterator)
            epoch += 1

        seq_len = input.size(1)

        if seq_len > max_seq_len:
            max_seq_len = seq_len

        pbar.set_description(
            f"LR: {optimizer.param_groups[0]['lr']:.3e} | Max Seq Len: {max_seq_len}"
        )

        input, target = input.to(device), target.to(device)

        with torch.autocast(
            device_type="cuda",
            dtype=(torch.bfloat16 if config.get("use_bf16", False) else torch.float32),
        ):

            _, loss = model(input, targets=target)

        # Adjust loss for gradient accumulation.
        loss = loss / gradient_accumulation_steps

        running_train_loss += loss.item() * gradient_accumulation_steps

        learning_rates.append(optimizer.param_groups[0]["lr"])

        loss = loss / gradient_accumulation_steps
        loss.backward()

        # Clip the gradients to a norm of 1.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if (step + 1) % gradient_accumulation_steps == 0:

            optimizer.step()

            optimizer.zero_grad()

        if "lr_scheduling" in config.keys():
            scheduler.step()

        tokens_processed += input.size(0) * input.size(1)

        # torch.cuda.empty_cache()
        # gc.collect()

        pbar.set_description(
            f"LR: {optimizer.param_groups[0]['lr']:.3e} | Max Seq Len: {max_seq_len}"
        )

        if config.get("print_memory_usage", False):

            print(
                f"Epoch {epoch} | Step {step} | Alloc: {torch.cuda.memory_allocated() / 1e9:.2f} GB | Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB "
            )

        if eval_freq and step % eval_freq == 0:
            torch.cuda.synchronize()

            train_loss = running_train_loss / (eval_freq if eval_freq <= step else 1)

            tokens_per_sec = tokens_processed / (time.time() - t0)

            running_train_loss = 0
            tokens_processed = 0
            t0 = time.time()

            with torch.no_grad():
                val_loss = compute_loss(model, val_dataloader, device)

            tqdm.write(
                f"Epoch {epoch} Global Step: {step} | Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f} | T/s {tokens_per_sec:.2f}"
            )

            # Log to ClearML
            if config["log_to_clearml"]:
                logger = Task.current_task().get_logger()

                logger.report_scalar(
                    "Training", "train_loss", iteration=step, value=train_loss
                )
                logger.report_scalar(
                    "Training", "val_loss", iteration=step, value=val_loss
                )

        if generation_freq and step % generation_freq == 0:

            print("#######################################################")
            print(f"#### GENERATING RESPONSES FOR EPOCH {epoch} STEP {step} ########")
            print("#######################################################")

            for entry in test_data[:3]:
                prompt = format_w_alpaca(entry)

                generate_response(model, tokenizer, prompt, get_device())

            print("#######################################################")

            # if config["log_to_clearml"]:
            #     logger.report_text("model_generation", model_generation, step=step)

    return model


# Write a function to download the data using clearml storaga manager.
def load_data(data_path: str):
    """
    Download the data from the provided link and store it in the specified path.

    Parameters
    ----------
    data_path : str
        The path where the data will be stored.
    """

    # Since the file is a json file, we use json to read the object.
    data = json.load(open(data_path))

    return data


def split_data(dataset, train_size=0.9, test_size=0.05):
    """
    Split the dataset into training, testing, and validation sets.

    Parameters
    ----------
    dataset : list
        The dataset to be split.
    train_size : float, optional
        The proportion of the dataset to include in the training set (default is 0.9).
    test_size : float, optional
        The proportion of the dataset to include in the testing set (default is 0.05).

    Returns
    -------
    train_data : list
        The training set.
    test_data : list
        The testing set.
    val_data : list
        The validation set.
    """

    train_portion = int(len(dataset) * train_size)
    test_portion = int(len(dataset) * test_size)

    train_data = dataset[:train_portion]
    test_data = dataset[train_portion : train_portion + test_portion]
    val_data = dataset[train_portion + test_portion :]

    return train_data, test_data, val_data


def create_dataloaders(train_data, test_data, val_data, tokenizer, config):
    """
    Create dataloaders for the training, testing, and validation sets.

    Parameters
    ----------
    train_data : list
        The training set.
    test_data : list
        The testing set.
    val_data : list
        The validation set.
    tokenizer : object
        The tokenizer object to encode the data.
    config : dict
        The configuration dictionary.

    Returns
    -------
    train_dataloader : torch.utils.data.DataLoader
        The dataloader for the training set.
    test_dataloader : torch.utils.data.DataLoader
        The dataloader for the testing set.
    val_dataloader : torch.utils.data.DataLoader
        The dataloader for the validation set.
    """

    torch.manual_seed(config["seed"])

    eval_batch_size = config["eval_batch_size"]
    batch_size = config["batch_size"] // config.get("gradient_accumulation_steps", 1)

    collate_fn = functools.partial(
        custom_collate_fn,
        allowed_max_length=config["max_seq_len"],
    )

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return {
        "train": train_dataloader,
        "test": test_dataloader,
        "val": val_dataloader,
    }


def load_tokenizer():
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


def create_model(config):

    model_cfg = model_configs[config["foundation_model"]]

    model = Llama3(model_cfg)

    weights_file = hf_hub_download(
        repo_id=model_cfg["hf_load_info"]["repo_id"],
        filename=model_cfg["hf_load_info"]["filename"],
        local_dir=config["foundation_model"],
    )

    weights = load_file(weights_file)

    load_hf_weights_into_llama(model, model_cfg, weights)

    del weights

    model = model.to(config["device"]).eval()

    # if enable_lora:
    #     # Freeze the parameters of the model.
    #     for param in model.parameters():
    #         param.requires_grad = False

    #     # Replace the linear layers with LinearWithLora layers.
    #     replace_linear_with_lora(model, 8, 16)

    #     num_trainable_params = sum(
    #         p.numel() for p in model.parameters() if p.requires_grad
    #     )

    #     print(f"Number of trainable parameters: {num_trainable_params}")

    # if preload:
    #     with open(GPT_ROOT / config["model_save_path"], "rb") as f:
    #         model.load_state_dict(torch.load(f))

    return model


def finetune_llama(config, model, dataloaders, tokenizer, test_data):

    if config["log_to_clearml"]:

        task = Task.init(
            project_name=config["project_name"],
            task_name=config["experiment_name"],
            task_type="training",
        )

        task.connect(flatten_dict(config))

    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]

    if config.get("use_explicit_bfloat16", False):
        model = model.to(torch.bfloat16)

    ########################################################################
    ########################################################################
    ################## FINETUNE THE FOUNDATION MODEL #######################
    ########################################################################
    ########################################################################

    torch.manual_seed(config["seed"])

    model = llama_trainer(
        config,
        model,
        train_dataloader,
        val_dataloader,
        tokenizer,
        test_data,
        # num_epochs=config["num_epochs"],
        num_train_iters=config["num_train_iters"],
        eval_freq=config["eval_freq"],
        generation_freq=config["generation_freq"],
        device=get_device(),
    )

    # Save the weights
    with open(GPT_ROOT / config["model_save_path"], "wb") as f:
        torch.save(model.state_dict(), f)

    return model


def compute_results(config, model, tokenizer, test_data):

    model.enable_kv_caching()

    for entry in test_data[:3]:
        prompt = format_w_alpaca(entry)

        generate_response(model, tokenizer, prompt, get_device())

    # Generate the test data.
    for i, entry in tqdm(
        enumerate(test_data),
        total=len(test_data),
        desc="Generating responses",
        dynamic_ncols=True,
    ):
        test_data[i]["model_response"] = generate_response(
            model,
            tokenizer,
            format_w_alpaca(entry),
            "cuda" if torch.cuda.is_available() else "cpu",
            print_input=False,
        )[len(format_w_alpaca(entry)) :].strip()

    # Save the responses
    with open(GPT_ROOT / config["responses_save_path"], "w") as file:
        json.dump(test_data, file, indent=4)

    return test_data


def benchmark_responses(test_data):
    scores, feedbacks = evaluate_with_promestheus(test_data)

    return scores


if __name__ == "__main__":

    ##########################################################################
    #################### DEFINE THE EXPERIMENT CONFIG ########################
    ##########################################################################

    experiment_descriptor = "llama3.2-1B-ft-alpaca-70k-epochs"

    full_alpaca_config = FineTuningConfig(
        project_name="llama-instruction-finetuning",
        experiment_name=f"{experiment_descriptor}",
        data_path="data/alpaca_data_cleaned.json",
        max_seq_len=512,
        drop_rate=0.0,
        qkv_bias=True,
        device="cuda",
        foundation_model="llama_3_2_1B",
        tokenizer="gpt2",
        seed=100,
        lr=2.5e-5,
        batch_size=128,
        lr_scheduling={
            "init_lr": 0,
            "warmup_percentage": 0.04,
            "eta_min": 2.5e-6,
        },
        # num_epochs=2,
        num_train_iters=70000,
        eval_batch_size=4,
        gradient_accumulation_steps=128,
        eval_freq=5000,
        print_memory_usage=False,
        generation_freq=None,
        responses_save_path=f"results/responses_w_finetuned_{experiment_descriptor}.json",
        model_save_path=f"checkpoints/finetuned_{experiment_descriptor}.pth",
        log_to_clearml=False,
        enable_gradient_checkpointing=True,
        use_bf16=True,
        use_explicit_bfloat16=False,
        use_8bit_optim=True,
    )

    config = full_alpaca_config.__dict__

    # Download the data
    data = load_data(full_alpaca_config.data_path)

    # Split the data
    train_data, test_data, val_data = split_data(data)

    # Load the tokenizer
    tokenizer = load_tokenizer()

    # Create the dataloaders
    dataloaders = create_dataloaders(train_data, test_data, val_data, tokenizer, config)

    model = create_model(config)

    # # Finetune the model
    model = finetune_llama(config, model, dataloaders, tokenizer, test_data)

    # # Persist the model and results
    test_data = compute_results(config, model, tokenizer, test_data)

    # This frees up the space for prom-eval to run on the GPU.
    del model

    torch.cuda.empty_cache()
    gc.collect()

    #
    # # Benchmark the responses
    scores = benchmark_responses(test_data)
