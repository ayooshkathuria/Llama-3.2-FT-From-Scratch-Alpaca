import os
from llmlib.SFT.pipeline import SFTPipeline
from llmlib.utils.utils import FineTuningConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == "__main__":

    ##########################################################################
    #################### DEFINE THE EXPERIMENT CONFIG ########################
    ##########################################################################

    experiment_descriptor = "llama3.2-1B-ft-alpaca-lora"

    full_alpaca_config = FineTuningConfig(
        project_name="llama-instruction-finetuning",
        experiment_name=f"{experiment_descriptor}",
        data_path="data/alpaca_data_cleaned.json",
        max_seq_len=512,
        drop_rate=0.0,
        qkv_bias=True,
        device="cuda",
        foundation_model="llama_3_2_1B",
        enable_lora=True,
        seed=100,
        lr=2.5e-5,
        batch_size=128,
        lr_scheduling={
            "init_lr": 0,
            "warmup_percentage": 0.04,
            "eta_min": 2.5e-6,
        },
        # num_epochs=2,
        num_train_iters=12500,
        eval_batch_size=4,
        gradient_accumulation_steps=32,
        eval_freq=2000,
        print_memory_usage=False,
        generation_freq=None,
        responses_save_path=f"results/responses_w_finetuned_{experiment_descriptor}.json",
        model_save_path=f"checkpoints/finetuned_{experiment_descriptor}.pth",
        inference_only=True,
        preload_model=True,
        log_to_clearml=False,
        enable_gradient_checkpointing=True,
        use_bf16=True,
        use_explicit_bfloat16=False,
        use_8bit_optim=True,
    )

    pipeline = SFTPipeline(full_alpaca_config)

    scores = pipeline.run()

    # # Get the json with results.
    # json_file_path = "src/results/responses_w_finetuned_llama3.2-1B-ft-alpaca-lora.json"

    # import json

    # with open(json_file_path, "r") as f:
    #     results = json.load(f)

    # pipeline.benchmark_responses(results)
