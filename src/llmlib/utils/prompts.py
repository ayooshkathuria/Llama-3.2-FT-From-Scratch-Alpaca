def format_w_alpaca(entry: str, get_response: bool = False) -> str:
    """
    Formats an entry from the instruction dataset into a markdown string.

    Parameters
    ----------
    entry : str
        The entry from the instruction dataset.
    get_response : bool
        Whether to include the response in the formatted string.


    Returns
    -------
    str
        The formatted string.
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input", None) else ""

    response_heading = "\n\n### Response:"

    if get_response:
        response = f"\n{entry['output']}"

    else:
        response = ""

    return instruction_text + input_text + response_heading + response


JUDGE_SFT_PROMPT = """
You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance.
You will be given an instruction, a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing the evaluation criteria.
Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
Please do not generate any other opening, closing, and explanations.

Here is the rubric you should use to build your answer:
1: The response fails to address the instructions, providing irrelevant, incorrect, or excessively verbose information that detracts from the user's request.
2: The response partially addresses the instructions but includes significant inaccuracies, irrelevant details, or excessive elaboration that detracts from the main task.
3: The response follows the instructions with some minor inaccuracies or omissions. It is generally relevant and clear, but may include some unnecessary details or could be more concise.
4: The response adheres to the instructions, offering clear, accurate, and relevant information in a concise manner, with only occasional, minor instances of excessive detail or slight lack of clarity.
5: The response fully adheres to the instructions, providing a clear, accurate, and relevant answer in a concise and efficient manner. It addresses all aspects of the request without unnecessary details or elaboration

Provide your feedback as follows:

Feedback:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the instruction, the reference answer, and the response.

Instruction: {instruction}
Reference Answer: {reference}
Answer: {answer}


Provide your feedback. If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.
Feedback:::
Evaluation: """


VANILLA_PROMPT = (
    "Given the input `{instruction}` "
    "and correct output `{reference},` "
    "score the model response `{answer}`"
    f" on a scale from 0 to 100, where 100 is the best score. "
    f"Respond with the integer number only.  Do not include an explanation."
)
