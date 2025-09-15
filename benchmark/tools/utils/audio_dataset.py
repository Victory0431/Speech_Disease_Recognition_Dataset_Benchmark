# save as: utils/audio_dataset.py
import os
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


class AudioWindowDataset(Dataset):
    def __init__(self, file_list, labels,
                 mode="train", sample_rate=16000,
                 window_size=4096, window_stride=None,
                 return_file_id=False):
        """
        直接用 file_list 和 labels 初始化
        """
        if file_list is None or labels is None:
            raise ValueError("必须提供 file_list 和 labels")

        self.file_list = file_list
        self.labels = labels
        self.mode = mode
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride or window_size
        self.return_file_id = return_file_id

    @classmethod
    def from_root_dir(cls, root_dir, **kwargs):
        """
        从 root_dir 扫描目录生成 file_list 和 labels
        """
        file_list, labels = cls._collect_files(root_dir)
        return cls(file_list, labels, **kwargs)

    @staticmethod
    def _collect_files(root_dir):
        file_list, labels = [], []
        covid_dirs = [
            os.path.join(root_dir, "covid"),
            os.path.join(root_dir, "covid_mp3"),
        ]
        non_covid_dir = os.path.join(root_dir, "non_covid")

        # covid -> 1
        for cdir in covid_dirs:
            if os.path.exists(cdir):
                for f in os.listdir(cdir):
                    if f.lower().endswith(('.wav', '.mp3')):
                        file_list.append(os.path.join(cdir, f))
                        labels.append(1)

        # non_covid -> 0
        if os.path.exists(non_covid_dir):
            for f in os.listdir(non_covid_dir):
                if f.lower().endswith('.wav'):
                    file_list.append(os.path.join(non_covid_dir, f))
                    labels.append(0)

        if not file_list:
            raise ValueError(f"No audio files found in {root_dir}")
        return file_list, labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]

        wav, sr = torchaudio.load(file_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        wav = wav.squeeze(0)

        if self.mode == "train":
            # ---- 训练：随机取一个窗口 ----
            if wav.shape[0] > self.window_size:
                start = torch.randint(0, wav.shape[0] - self.window_size + 1, (1,)).item()
                wav = wav[start:start+self.window_size]
            else:
                wav = F.pad(wav, (0, self.window_size - wav.shape[0]))

            return wav, label  # shape [T], scalar

        else:
            # ---- 验证/测试：滑窗 ----
            segments = []
            for start in range(0, max(1, wav.shape[0] - self.window_size + 1), self.window_stride):
                seg = wav[start:start+self.window_size]
                if len(seg) < self.window_size:
                    seg = F.pad(seg, (0, self.window_size-len(seg)))
                segments.append(seg)

            segments = torch.stack(segments)  # [num_windows, T]

            if self.return_file_id:
                return segments, label, os.path.basename(file_path)
            return segments, label
