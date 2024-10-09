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
    audio_path = []
    spectrogram = []
    spectrogram_length = []
    text = []
    text_encoded = []
    text_encoded_length = []

    for item in dataset_items:
        audio_path.append(item["audio_path"])
        spectrogram.append(item["spectrogram"].squeeze(0).T)
        spectrogram_length.append(item["spectrogram"].shape[2])
        text.append(item["text"])
        text_encoded.append(item["text_encoded"].squeeze(0))
        text_encoded_length.append(item["text_encoded"].shape[1])

    return {
        "audio_path": audio_path,
        "spectrogram": pad_sequence(spectrogram, batch_first=True).transpose(1, 2).unsqueeze(1),
        "spectrogram_length": torch.tensor(spectrogram_length),
        "text": text,
        "text_encoded": pad_sequence(text_encoded, batch_first=True),
        "text_encoded_length": torch.tensor(text_encoded_length)
    }
