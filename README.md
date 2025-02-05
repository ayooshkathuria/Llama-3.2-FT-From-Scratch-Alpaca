# Llama 3.2 FT On Alpaca From Scratch

This repository houses my implementation of Llama 3.2Bfrom scratch. Currently, it just contains the ability to finetune Llama 3.2 (1B and 3B) variants on an Alpaca like finetuning dataset. In future, I intend to add RL based alignment techniques too. 

## Intstallation

The recommended way to setup the code is using `uv`, which is a modern python dependency management tool. 
Only python 3.10.x and 3.11.x are supported through UV. 

First download and set up `uv` and then you can clone the repo and run uv install. 

THe first thing you gotta do is to clone the folder and get inside the folder.

```
git clone 
cd Llama3.2-FT-Alpaca-From-Scratch
```

Now, we will setup a virtual env using `uv` and install the package. 

```python
uv venv    
source .venv/bin/activate 
uv pip install -e .  
```

If you want to install without uv, I recommended setting up a venv / conda env with Python 3.10/11 and 
then installing the package using `pip`

```
pip install -e .
```

## Finetuning

The code for finetuning resides in the `src/sft_pipeline.py` file. Currently, there is only support for Llama 3.2 1B / 3B and using alpaca style instruction datasets. 

The finetuning script has been designed with different components which will be modularised in the future. To run a finetuning job, we first have to define a `FineTuningConfig`
object which is then pass to the SFT function. This function is a dataclass. 

The following is a list of finetuning args. 


## Basic Configuration
| Parameter          | Type | Description                                                                                                                          |
| ------------------ | ---- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `project_name`     | str  | Name of the project                                                                                                                  |
| `experiment_name`  | str  | Name of the experiment (a random string will be appended)                                                                            |
| `data_path`        | str  | Path to the training data                                                                                                            |
| `foundation_model` | str  | Name/type of the base model to fine-tune. Available models can be found in the `model_configs` dictionary from `llmlib.utils.models` |
| `device`           | str  | Device to run training on ('cpu', 'cuda', etc.)                                                                                      |

## Model Architecture Parameters
| Parameter     | Type  | Default | Description                                       |
| ------------- | ----- | ------- | ------------------------------------------------- |
| `max_seq_len` | int   | -       | Maximum sequence length for input tokens          |
| `drop_rate`   | float | -       | Dropout rate to use in the model                  |
| `qkv_bias`    | bool  | -       | Whether to use bias in QKV attention calculations |

## Training Configuration
| Parameter                     | Type  | Default | Description                                                           |
| ----------------------------- | ----- | ------- | --------------------------------------------------------------------- |
| `seed`                        | int   | -       | Random seed for reproducibility                                       |
| `lr`                          | float | -       | Learning rate for training                                            |
| `lr_scheduling`               | dict  | -       | Learning rate scheduling configuration                                |
| `batch_size`                  | int   | -       | Training batch size                                                   |
| `weight_decay`                | float | 0.01    | Weight decay coefficient                                              |
| `num_epochs`                  | int   | None    | Number of training epochs (mutually exclusive with `num_train_iters`) |
| `num_train_iters`             | int   | None    | Number of training iterations (mutually exclusive with `num_epochs`)  |
| `gradient_accumulation_steps` | int   | 1       | Number of gradient accumulation steps                                 |

## Evaluation and Generation Settings
| Parameter             | Type | Default    | Description                                                 |
| --------------------- | ---- | ---------- | ----------------------------------------------------------- |
| `eval_batch_size`     | int  | None       | Batch size for evaluation (defaults to training batch_size) |
| `eval_freq`           | int  | None       | Frequency of evaluation in iterations                       |
| `max_gen_tokens`      | int  | 256        | Maximum number of tokens to generate during inference       |
| `generation_freq`     | int  | None       | Frequency of text generation in iterations                  |
| `use_ollama_for_eval` | bool | False      | Whether to use Ollama for evaluation                        |
| `ollama_model_name`   | str  | 'llama3.1' | Name of the Ollama model to use for evaluation              |

## Optimization and Memory Settings
| Parameter                       | Type | Default | Description                                      |
| ------------------------------- | ---- | ------- | ------------------------------------------------ |
| `enable_gradient_checkpointing` | bool | False   | Whether to enable gradient checkpointing         |
| `use_bf16`                      | bool | False   | Whether to use bfloat16 precision for inference. |
| `use_explicit_bfloat16`         | bool | False   | Whether to explicitly use bfloat16 dtype         |
| `use_8bit_optim`                | bool | False   | Whether to use 8-bit AdamW optimiser.            |
| `print_memory_usage`            | bool | False   | Whether to print memory usage statistics         |

## Model and Output Management
| Parameter             | Type | Default | Description                                                                                                     |
| --------------------- | ---- | ------- | --------------------------------------------------------------------------------------------------------------- |
| `inference_only`      | bool | False   | Whether to run in inference-only mode                                                                           |
| `preload_model`       | bool | False   | Whether to preload the model. In this case, the model will be loaded from the path defined by `model_save_path` |
| `enable_lora`         | bool | False   | Whether to enable LoRA fine-tuning                                                                              |
| `responses_save_path` | str  | None    | Path to save generated responses                                                                                |
| `model_save_path`     | str  | None    | Path to save model checkpoints                                                                                  |
| `log_to_clearml`      | bool | False   | Whether to log metrics to ClearML                                                                               |

## Important Notes
- Either `num_train_iters` or `num_epochs` must be provided, but not both
- If `lr_scheduling` is provided, `init_lr` and `eta_min` will default to `lr` if not specified
- Model-specific configurations from `model_configs` will be added to the instance attributes
- `batch_size` is the effective batch size and if `gradient_accumulation_steps` > 0, the batch size processed by the LLM becomes (`batch_size / gradient_accumulation_steps`) with gradient updates every `gradient_accumulation_steps` steps. 


This config is then sed to `run_sft_pipeline` to train a function. The plots are saved to a clearml server pointed by the clearml key. 



For example, how to run a finetuneing job is shown below.

```python
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
        seed=100,
        lr=2.5e-5,
        batch_size=128,
        lr_scheduling={
            "init_lr": 0,
            "warmup_percentage": 0.04,
            "eta_min": 2.5e-6,
        },
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

    run_sft_pipeline(full_alpaca_config)
```

All the paths, when not absolute, are relative to the root of the project. 

## Train Plots

There are three training scripts available for now. One that finetunes, Llama 3.2 1B, Llama 3.2 1B with LoRa and Llama 3.2 3B with Lora. Te first two were trained with a max sequence length of 512 and the other with 256. This is what I could fit in my 24 GB GPU. 


### Evaluation

We use LLM-as-a-Judge as a way to evaluate the output of the instructio fine-funed model. There are two ways you can go about it. 

1. Using `Prometheus-eval`. This was an LLM that was specifically finetuned to rate instruction following tasks and has an equivalent performance. It comes in two variants an 8x7B MoE model and a lighter 7B model that is on par with Mistral 8x7B for evaluation. I have been using the 7B model for evaluation as it takes around 16 GB of memory. This is what is used by default in the project. The finetuned models responses are logged to the disk and then the finetuned model is deleted from 
the GPU VRAM to make sure that the model can be run on a 24 GB GPU. 

2. Using Ollama. If you don't have a > 16 GB GPU, you can use Ollama to evaluate the model. This uses a custom custom prompt that is defined in the file `src/llmlib/utils/prompts.py`. You can find the prompt called `JUDGE_SFT_PROMPT` that is used in this file. You can also use this when you want to try a new evaluator model that is available through Ollama. 

The evaluation function parses the response of the evaluator LLM but sometimes, it may fail to get a response and those scores are excluded from the final score computation. The LLM are given a rubric of scoring and asked to score the response on a scale of 1-5. The scores are then averaged to get the final score. \

The following rubric is used to score the answers. 

```
1: The response fails to address the instructions, providing irrelevant, incorrect, or excessively verbose information that detracts from the user's request.
2: The response partially addresses the instructions but includes significant inaccuracies, irrelevant details, or excessive elaboration that detracts from the main task.
3: The response follows the instructions with some minor inaccuracies or omissions. It is generally relevant and clear, but may include some unnecessary details or could be more concise.
4: The response adheres to the instructions, offering clear, accurate, and relevant information in a concise manner, with only occasional, minor instances of excessive detail or slight lack of clarity.
5: The response fully adheres to the instructions, providing a clear, accurate, and relevant answer in a concise and efficient manner. It addresses all aspects of the request without unnecessary details or elaboration

```

Here are the results for each of three finetuning experiments. 

| Model                | Score   |
| -------------------- | ------- |
| Llama 3.2 1B         | 3.41    |
| Llama 3.2 1B w/ LoRA | 3.23    |
| Llama 3.2 3B w/ LoRA | **3.5** |


### Contribution

The code is extremely limited in scope and I would love to have contributions to the code. Please feel free to open a PR and I will be happy to review it.

### Acknowledgements

The follwoing resources have been inspirations for this work. 

1. [Let's Reproduce GPT-2 (124M) by Andej Karpathy](https://www.youtube.com/watch?v=l8pRSuU81PU)
2. [Build a Large Language Model (From Scratch) by Sebastian Raschka](https://www.manning.com/books/build-a-large-language-model-from-scratch)

