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

    # max_length = max(sample['input_wav'].shape[1] for sample in dataset_items)
    max_length = 25600

    padded_batch = []
    for sample in dataset_items:
        pad_size = max_length - sample["input_wav"].shape[1]
        padded_sample = torch.nn.functional.pad(
            sample["input_wav"], (0, pad_size), "constant", 0
        )
        padded_batch.append(padded_sample)

    return {"input": torch.stack(padded_batch).squeeze(1)}
