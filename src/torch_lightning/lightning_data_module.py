import os
import random

import torch
import librosa
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from src.transforms.spectrogram import MelSpectrogramTransform
from src.data_download.download_drone_audio import download_and_extract_drone_audio_dataset


class DroneAudioDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        waveform, sr = librosa.load(audio_path, sr=None, mono=False)

        if self.transform is not None:
            x = self.transform(waveform, sr)
        else:
            if waveform.ndim == 1:
                x = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
            else:
                x = torch.tensor(waveform, dtype=torch.float32)

        y = torch.tensor(label, dtype=torch.long)
        return x, y


class DroneDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir="data",
        dataset_type="binary",   # "binary" albo "multiclass"
        variant="original",      # "original", "noisy", "kalman", "noisy_kalman"
        batch_size=16,
        num_workers=0,
        sample_rate=16000,
        duration=2.0,
        n_mels=128,
        n_fft=1024,
        hop_length=256,
        noise_std=0.005,
        kalman_q=1e-5,
        kalman_r=1e-3,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    ):
        super().__init__()

        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.variant = variant

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.noise_std = noise_std
        self.kalman_q = kalman_q
        self.kalman_r = kalman_r

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        self.data_dir = None
        self.meta = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.class_to_idx = None
        self.idx_to_class = None

    def prepare_data(self):
        meta = download_and_extract_drone_audio_dataset(self.root_dir)

        if self.dataset_type == "binary":
            self.data_dir = meta.binary_dir
        elif self.dataset_type == "multiclass":
            self.data_dir = meta.multiclass_dir
        else:
            raise ValueError("dataset_type must be 'binary' or 'multiclass'.")

        self.meta = meta

    def _collect_files(self):
        if self.data_dir is None:
            raise ValueError("data_dir is None. Call prepare_data() or setup() first.")

        class_names = []

        for name in os.listdir(self.data_dir):
            full_path = os.path.join(self.data_dir, name)
            if os.path.isdir(full_path):
                class_names.append(name)

        class_names.sort()

        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
        self.idx_to_class = {i: class_name for class_name, i in self.class_to_idx.items()}

        file_paths = []
        labels = []

        for class_name in class_names:
            class_dir = os.path.join(self.data_dir, class_name)

            for root, _, files in os.walk(class_dir):
                for file_name in files:
                    lower_name = file_name.lower()
                    if lower_name.endswith((".wav", ".mp3", ".flac", ".ogg", ".au")):
                        full_path = os.path.join(root, file_name)
                        file_paths.append(full_path)
                        labels.append(self.class_to_idx[class_name])

        return file_paths, labels

    def setup(self, stage=None):
        if self.data_dir is None:
            self.prepare_data()

        file_paths, labels = self._collect_files()

        data = list(zip(file_paths, labels))
        random.seed(self.seed)
        random.shuffle(data)

        total_size = len(data)

        train_end = int(total_size * self.train_ratio)
        val_end = train_end + int(total_size * self.val_ratio)

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        train_paths = [x[0] for x in train_data]
        train_labels = [x[1] for x in train_data]

        val_paths = [x[0] for x in val_data]
        val_labels = [x[1] for x in val_data]

        test_paths = [x[0] for x in test_data]
        test_labels = [x[1] for x in test_data]

        transform = MelSpectrogramTransform(
            sample_rate=self.sample_rate,
            duration=self.duration,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            variant=self.variant,
            noise_std=self.noise_std,
            kalman_q=self.kalman_q,
            kalman_r=self.kalman_r,
        )

        self.train_dataset = DroneAudioDataset(train_paths, train_labels, transform=transform)
        self.val_dataset = DroneAudioDataset(val_paths, val_labels, transform=transform)
        self.test_dataset = DroneAudioDataset(test_paths, test_labels, transform=transform)

        labels = torch.tensor(self.train_labels)
        class_counts = torch.bincount(labels)  # [count_0, count_1]
        class_weights = 1. / class_counts.float()
        sample_weights = class_weights[labels]

        self.train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        print("=" * 60)
        print("Dataset root:", self.root_dir)
        print("Selected data dir:", self.data_dir)
        print("Dataset type:", self.dataset_type)
        print("Variant:", self.variant)
        print("Class to idx:", self.class_to_idx)
        print("Train size:", len(self.train_dataset))
        print("Val size:", len(self.val_dataset))
        print("Test size:", len(self.test_dataset))
        print("=" * 60)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )