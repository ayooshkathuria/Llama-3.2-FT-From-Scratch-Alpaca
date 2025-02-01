import re

import torch
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
from prometheus_eval.vllm import VLLM
from tqdm import tqdm

from llmlib.utils.models import query_ollama_model
from llmlib.utils.prompts import JUDGE_SFT_PROMPT, format_w_alpaca


def generate_response(gpt_model, tokenizer, prompt, device, print_input=True):

    prompt_tk_ids = torch.Tensor(tokenizer.encode(prompt)).to(device)

    generated_text = gpt_model.generate(
        max_new_token=256,
        inp_tk_ids=prompt_tk_ids,
        context_length=256,
        eos_id=[128001],
    )

    model_reponse = tokenizer.decode(generated_text[0].tolist())

    if print_input:

        print(prompt)

        print("\nModel Generation >>")

        print(model_reponse[len(prompt) :].strip())
        print("-" * 50)

    return model_reponse


def extract_judge_score(answer: str, split_str: str = "Total rating:") -> int:
    try:
        if split_str in answer:
            rating = answer.split(split_str)[1]
        else:
            rating = answer
        digit_groups = [el.strip() for el in re.findall(r"\d+(?:\.\d+)?", rating)]
        return float(digit_groups[0])
    except Exception as e:
        print(e)
        return None


def evaluate_w_ollama(test_data, model_name="llama3.1", prompt=JUDGE_SFT_PROMPT):
    scores = []
    for i, entry in tqdm(enumerate(test_data), total=len(test_data), desc="Evaluating"):

        prompt_to_evaluator_llm = prompt.format(
            instruction=format_w_alpaca(entry),
            reference=entry["output"],
            answer=test_data[i]["model_response"],
        )

        answer = query_ollama_model(prompt_to_evaluator_llm, model_name)

        # try:
        #     score = int(answer)
        #     scores.append(score)
        # except ValueError:
        #     # print(f"Could not convert score: {answer}")
        #     score = None

        score = extract_judge_score(answer)

        if score is not None:
            test_data[i]["model_score"] = int(score)
            scores.append(int(score))
        else:
            print(f"Could not convert score: {score}")
            test_data[i]["model_score"] = None
            continue

    print("Could compute scores for ", len(scores), " entries.")
    print(f"The average score is {sum(scores) / len(scores):.2f}")

    return scores, []


def evaluate_with_promestheus(test_data):
    # Load the model.
    model = VLLM(model="prometheus-eval/prometheus-7b-v2.0", max_model_len=2048)
    judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

    # rubric_data = {
    #     "criteria": "Does the response adhere to the instructions, demonstrating clarity, relevance, correctness, and conciseness?",
    #     "score1_description": "The response fails to address the instructions, providing irrelevant, incorrect, or excessively verbose information that detracts from the user's request.",
    #     "score2_description": "The response partially addresses the instructions but includes significant inaccuracies, irrelevant details, or excessive elaboration that detracts from the main task.",
    #     "score3_description": "The response follows the instructions with some minor inaccuracies or omissions. It is generally relevant and clear, but may include some unnecessary details or could be more concise.",
    #     "score4_description": "The response adheres to the instructions, offering clear, accurate, and relevant information in a concise manner, with only occasional, minor instances of excessive detail or slight lack of clarity.",
    #     "score5_description": "The response fully adheres to the instructions, providing a clear, accurate, and relevant answer in a concise and efficient manner. It addresses all aspects of the request without unnecessary details or elaboration.",
    # }

    rubric_data = {
        "criteria": "Ability to follow the instruction while providing semantically valid and relevant information without being penalized for deviating from the reference response.",
        "score1_description": "Response is nonsensical or entirely unrelated to the instruction.",
        "score2_description": "Response attempts to follow the instruction but includes major irrelevance or misunderstanding.",
        "score3_description": "Response follows the instruction but has minor errors or unnecessary details.",
        "score4_description": "Response is semantically valid and adheres to the instruction with slight stylistic or verbosity flaws.",
        "score5_description": "Response perfectly follows the instruction with concise, accurate, and relevant information.",
    }

    score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

    instructions = [format_w_alpaca(x) for x in test_data]
    responses = [x["model_response"] for x in test_data]
    reference_answers = [x["output"] for x in test_data]

    feedbacks, scores = judge.absolute_grade(
        instructions=instructions,
        responses=responses,
        rubric=score_rubric,
        reference_answers=reference_answers,
    )

    scores = [int(score) for score in scores if score is not None]

    print("Could compute scores for ", len(scores), " entries.")
    print(f"The average score is {sum(scores) / len(scores):.2f}")

    return scores, feedbacks


if __name__ == "__main__":
    # import json

    # from llmlib import GPT_ROOT

    # test_data_1 = json.load(open(GPT_ROOT / "data/responses_w_pretrained_model.json"))

    # test_data_2 = json.load(open(GPT_ROOT / "data/responses_w_finetuned_model.json"))

    # test_data_3 = json.load(open(GPT_ROOT / "data/responses_with_gt.json"))

    # model_name = "llama3.1"

    # evaluate_w_ollama(test_data_1, model_name=model_name, prompt=JUDGE_SFT_PROMPT)

    # evaluate_w_ollama(test_data_2, model_name=model_name, prompt=JUDGE_SFT_PROMPT)

    # evaluate_w_ollama(test_data_3, model_name=model_name, prompt=JUDGE_SFT_PROMPT)

    test_data_path = "data/responses_w_finetuned_gpt2-xl-alpaca.json"
    import json

    test_data = json.load(open(test_data_path))

    scores, feedbacks = evaluate_with_promestheus(test_data)
