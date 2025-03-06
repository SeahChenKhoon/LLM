from datasets import load_dataset
from datasets import DatasetDict, load_dataset
from typing import List, Dict, Tuple

DASH_LINE = "-" * 100  # Create a 100-character long separator line


def huggingface_dataset() -> DatasetDict:
    """
    Loads the 'knkarthick/dialogsum' dataset from Hugging Face.

    Returns:
        DatasetDict: A dictionary-like object containing the dataset splits.
    """
    huggingface_dataset_name: str = "knkarthick/dialogsum"
    return load_dataset(huggingface_dataset_name)


def display_dialogue_summaries(
    sample_indices: List[int], dataset: Dict[str, Dict[str, List[Dict[str, str]]]]
) -> None:
    """
    Displays dialogues and corresponding human-generated summaries for the given sample indices.

    Args:
        sample_indices (List[int]): A list of indices corresponding to examples in the dataset.
        dataset (Dict[str, Dict[str, List[Dict[str, str]]]]): A nested dictionary containing test
        dialogues and summaries.

    Raises:
        KeyError: If the dataset structure is incorrect or an index is missing.
    """
    for i, index in enumerate(sample_indices):
        print(DASH_LINE)
        print(f"Example {i + 1}")
        print(DASH_LINE)
        print("INPUT DIALOGUE:")
        print(dataset["test"][index]["dialogue"])
        print(DASH_LINE)
        print("BASELINE HUMAN SUMMARY:")
        print(dataset["test"][index]["summary"])
        print(DASH_LINE)
        print()


def output_text(task: str, summary: str, output: str) -> None:
    """
    Displays the baseline human summary and the model-generated output.

    Args:
        task (str): The name of the task (e.g., "ZERO SHOT", "FEW SHOT").
        summary (str): The ground-truth human-written summary.
        output (Any): The model-generated summary or text.
    """
    print(DASH_LINE)
    print(f"BASELINE HUMAN SUMMARY:\n{summary}\n")
    print(DASH_LINE)
    print(f"MODEL GENERATION - {task}:\n{output}\n")


def make_prompt(
    dataset: Dict[str, Dict[str, List[Dict[str, str]]]],
    instruction: str,
    index_to_summarize: int,
    list_index_to_train: List[int],
) -> Tuple[str, str]:
    """
    Constructs a prompt for a model, including few-shot examples (if provided) and the main input.

    Args:
        dataset (Dict[str, Dict[str, List[Dict[str, str]]]]): The dataset containing dialogues and summaries.
        instruction (str): The instruction given to the model (e.g., "Summarize the conversation").
        index_to_summarize (int): The index of the dialogue to summarize.
        list_index_to_train (List[int]): Indices for few-shot training examples.

    Returns:
        Tuple[str, str]: The generated prompt and the corresponding ground-truth summary.
    """
    prompt = ""

    # Add few-shot training examples (if provided)
    for index in list_index_to_train:
        dialogue = dataset["test"][index]["dialogue"]
        summary = dataset["test"][index]["summary"]

        # The stop sequence '{summary}\n\n\n' is important for FLAN-T5. Adjust for other models as needed.
        prompt += f"""
Dialogue:

{dialogue}

{instruction}
{summary}
"""

    # Add the main dialogue to summarize
    dialogue = dataset["test"][index_to_summarize]["dialogue"]
    summary = dataset["test"][index_to_summarize]["summary"]

    prompt += f"""
Dialogue:

{dialogue}

{instruction}
"""

    return prompt, summary
