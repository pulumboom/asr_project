from pathlib import Path

from src.datasets.base_dataset import BaseDataset

import torchaudio


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        data = []
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)
                if transcription_dir and Path(transcription_dir).exists():
                    transc_path = Path(transcription_dir) / (path.stem + ".txt")
                    if transc_path.exists():
                        with transc_path.open() as f:
                            entry["text"] = f.read().strip().lower()
                t_info = torchaudio.info(str(path))
                length = t_info.num_frames / t_info.sample_rate
                entry["audio_len"] = length
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
