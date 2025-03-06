from datasets import load_dataset
DASH_LINE = '-' * 100  # Create a 100-character long separator line

def huggingface_dataset():
    """
    Load a dataset from Hugging Face.

    Args:
    dataset_name (str): The name of the dataset to load.

    Returns:
    DatasetDict: The loaded dataset.
    """
    huggingface_dataset_name = "knkarthick/dialogsum"
    return load_dataset(huggingface_dataset_name)
    

def display_dialogue_summaries(sample_indices: list, dataset: dict) -> None:
    """
    Displays dialogue and corresponding human-generated summaries for the given sample indices.

    Args:
        sample_indices (list): List of indices corresponding to examples in the dataset.
        dataset (dict): The dataset containing test dialogues and summaries.
    """
    for i, index in enumerate(sample_indices):
        print(DASH_LINE)
        print(f'Example {i + 1}')
        print(DASH_LINE)
        print('INPUT DIALOGUE:')
        print(dataset['test'][index]['dialogue'])
        print(DASH_LINE)
        print('BASELINE HUMAN SUMMARY:')
        print(dataset['test'][index]['summary'])
        print(DASH_LINE)
        print()  # Add a blank line for better readability


def output_text(task, summary, output):
    print(DASH_LINE)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
    print(DASH_LINE)
    print(f'MODEL GENERATION - {task}:\n{output}\n')


def make_prompt(dataset, instruction, index_to_summarize, list_index_to_train):
    prompt = ''
    for index in list_index_to_train:
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']

        # The stop sequence '{summary}\n\n\n' is important for FLAN-T5. Other models may have their own preferred stop sequence.
        prompt += f"""
            Dialogue:

            {dialogue}

            {instruction}
            {summary}
        """

    dialogue = dataset['test'][index_to_summarize]['dialogue']
    summary = dataset['test'][index_to_summarize]['summary']

    prompt += f"""
Dialogue:

{dialogue}

{instruction}
"""

    return prompt, summary