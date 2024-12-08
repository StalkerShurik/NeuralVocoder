import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    if "input_text" in list(dataset_items[0].keys()):
        return {"input_text": dataset_items[0]["input_text"]}

    if len(dataset_items) == 1:
        return {"input": dataset_items[0]["input_wav"]}

    max_length = 25600

    padded_batch = []
    for sample in dataset_items:
        pad_size = max_length - sample["input_wav"].shape[1]
        padded_sample = torch.nn.functional.pad(
            sample["input_wav"], (0, pad_size), "constant", 0
        )
        padded_batch.append(padded_sample)

    return {"input": torch.stack(padded_batch).squeeze(1)}
