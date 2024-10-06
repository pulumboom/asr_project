import torch
from torch.nn.utils.rnn import pad_sequence


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
    if len(dataset_items) == 0:
        return {}

    result_batch = {}
    for key in dataset_items[0].keys():
        if key == "text_encoded":
            result_batch[key] = pad_sequence(
                [elem[key].T for elem in dataset_items],
                batch_first=True
            ).squeeze()
        elif key == "spectrogram":
            result_batch[key] = pad_sequence(
                [elem[key].T for elem in dataset_items],
                batch_first=True
            ).squeeze().transpose(-1, -2)
        elif isinstance(dataset_items[0][key], torch.Tensor):
            result_batch[key] = pad_sequence(
                [elem[key].T for elem in dataset_items],
                batch_first=True
            ).permute(0, *range(dataset_items[0][key].dim(), 0, -1))
        else:
            result_batch[key] = [
                elem[key] for elem in dataset_items
            ]
    result_batch["spectrogram_length"] = torch.tensor([
        item["spectrogram"].shape[2] for item in dataset_items
    ])
    result_batch["text_encoded_length"] = torch.tensor([
        item["text_encoded"].shape[1] for item in dataset_items
    ])

    return result_batch
