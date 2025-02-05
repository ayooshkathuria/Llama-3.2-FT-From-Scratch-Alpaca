import gc
import json

import torch
from clearml import Task
from tqdm import tqdm

from llmlib import GPT_ROOT
from llmlib.Llama.llama import Llama3, create_llama3_model, replace_lora_with_linear
from llmlib.Llama.preloaded_llama import load_hf_weights_into_llama
from llmlib.Llama.tokenizers import Llama3Tokenizer, load_llama3_tokenizer
from llmlib.SFT.train import create_dataloaders, sft_trainer
from llmlib.utils.eval import (
    evaluate_w_ollama,
    evaluate_with_promestheus,
    generate_response,
)
from llmlib.utils.lr_schedulers import WarmupCosineAnnealingLR
from llmlib.utils.models import get_device
from llmlib.utils.prompts import format_w_alpaca
from llmlib.utils.utils import FineTuningConfig, flatten_dict


class SFTPipeline:
    def __init__(self, config):
        self.config = config.__dict__

    def load_data(self, data_path: str):
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

    def split_data(self, dataset, train_size=0.9, test_size=0.05):
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

    def finetune(self, model, dataloaders, tokenizer, test_data):

        config = self.config

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

        model = sft_trainer(
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

        if config.get("enable_lora", False):
            replace_lora_with_linear(model)

        # Save the weights
        (GPT_ROOT / config["model_save_path"]).parent.mkdir(parents=True, exist_ok=True)
        with open(GPT_ROOT / config["model_save_path"], "wb") as f:
            torch.save(model.state_dict(), f)

        return model

    def compute_results(self, model, tokenizer, test_data):

        config = self.config

        model.enable_kv_caching()

        for entry in test_data[:3]:
            prompt = format_w_alpaca(entry)

            generate_response(
                model,
                tokenizer,
                prompt,
                get_device(),
                max_new_tokens=config["max_gen_tokens"],
                context_length=config["max_seq_len"],
            )

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
                context_length=config["max_seq_len"],
            )[len(format_w_alpaca(entry)) :].strip()

        # Save the responses
        (GPT_ROOT / config["responses_save_path"]).parent.mkdir(
            parents=True, exist_ok=True
        )
        with open(GPT_ROOT / config["responses_save_path"], "w") as file:
            json.dump(test_data, file, indent=4)

        # Log file file to clearml if enabled
        if config.get("log_to_clearml", False):
            task = Task.current_task()
            task.upload_artifact(
                name="Results", artifact_object=GPT_ROOT / config["responses_save_path"]
            )

        return test_data

    def benchmark_responses(self, test_data):
        config = self.config
        if not config.get("use_ollama_for_eval", False):
            scores, feedbacks = evaluate_with_promestheus(test_data)
        else:
            scores, feedbacks = evaluate_w_ollama(
                test_data, model_name=config["ollama_model_name"]
            )

        avg_score = sum(scores) / len(scores)

        if config.get("log_to_clearml", False):
            logger = Task.current_task().get_logger()

            logger.report_scalar(
                "Evaluation", "Avg Score", iteration=0, value=avg_score
            )
        return scores

    def run(self):
        config = self.config

        # Download the data
        data = self.load_data(config["data_path"])

        # Split the data
        train_data, test_data, val_data = self.split_data(data)

        # Load the tokenizer
        tokenizer = load_llama3_tokenizer()

        # Create the dataloaders
        dataloaders = create_dataloaders(
            train_data, test_data, val_data, tokenizer, config
        )

        model = create_llama3_model(config)

        # # Finetune the model
        if not config.get("inference_only", False):
            model = self.finetune(model, dataloaders, tokenizer, test_data)

        # # Persist the model and results
        test_data = self.compute_results(model, tokenizer, test_data)

        # This frees up the space for prom-eval to run on the GPU.
        del model

        torch.cuda.empty_cache()
        gc.collect()

        #
        # # Benchmark the responses
        scores = self.benchmark_responses(test_data)

        return scores
