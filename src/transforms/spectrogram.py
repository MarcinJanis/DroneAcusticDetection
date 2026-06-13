import numpy as np
import torch
import librosa


class MelSpectrogramTransform:
    def __init__(
        self,
        sample_rate=16000,
        duration=2.0,
        n_mels=128,
        n_fft=1024,
        hop_length=256,
        f_min=0.0,
        f_max=None,
        power=2.0,
        variant="original",  # "original", "noisy", "noisy_kalman"
        noise_std=0.01,
        kalman_q=1e-5,
        kalman_r=1e-3,
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.power = power

        self.variant = variant
        self.noise_std = noise_std
        self.kalman_q = kalman_q
        self.kalman_r = kalman_r

        self.target_num_samples = int(sample_rate * duration)

        allowed = {"original", "noisy", "kalman", "noisy_kalman"}
        if self.variant not in allowed:
            raise ValueError(f"variant must be one of {allowed}, got {self.variant}")

    def to_mono(self, waveform):
        if waveform.ndim == 1:
            return waveform.astype(np.float32)

        if waveform.ndim == 2:
            return np.mean(waveform, axis=0).astype(np.float32)

        raise ValueError(f"Unsupported waveform shape: {waveform.shape}")

    def resample_if_needed(self, waveform, sr):
        if sr != self.sample_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=sr,
                target_sr=self.sample_rate,
            )
            sr = self.sample_rate

        return waveform.astype(np.float32), sr

    def pad_or_trim(self, waveform):
        current_num_samples = waveform.shape[0]

        if current_num_samples > self.target_num_samples:
            waveform = waveform[:self.target_num_samples]
        elif current_num_samples < self.target_num_samples:
            pad_amount = self.target_num_samples - current_num_samples
            waveform = np.pad(waveform, (0, pad_amount), mode="constant")

        return waveform.astype(np.float32)

    def add_gaussian_noise(self, waveform):
        noise = np.random.normal(0.0, self.noise_std, size=waveform.shape).astype(np.float32)
        noisy = waveform.astype(np.float32) + noise
        noisy = np.clip(noisy, -1.0, 1.0)
        return noisy.astype(np.float32)

    def kalman_filter_1d(self, waveform):

        waveform = waveform.astype(np.float32)
        filtered = np.zeros_like(waveform, dtype=np.float32)

        if len(waveform) == 0:
            return filtered

        x_est = float(waveform[0])
        p = 1.0
        q = float(self.kalman_q)
        r = float(self.kalman_r)

        for k in range(len(waveform)):
            z = float(waveform[k])

            x_pred = x_est
            p_pred = p + q

            k_gain = p_pred / (p_pred + r)
            x_est = x_pred + k_gain * (z - x_pred)
            p = (1.0 - k_gain) * p_pred

            filtered[k] = x_est

        return filtered.astype(np.float32)

    def apply_variant(self, waveform):
        waveform = waveform.astype(np.float32)

        if self.variant == "original":
            return waveform

        if self.variant == "noisy":
            return self.add_gaussian_noise(waveform)

        if self.variant == "kalman":
            return self.kalman_filter_1d(waveform)

        if self.variant == "noisy_kalman":
            waveform = self.add_gaussian_noise(waveform)
            waveform = self.kalman_filter_1d(waveform)
            return waveform

        return waveform

    def __call__(self, waveform, sr):
        waveform = self.to_mono(waveform)
        waveform, sr = self.resample_if_needed(waveform, sr)
        waveform = self.pad_or_trim(waveform)

        waveform = self.apply_variant(waveform)

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            power=self.power,
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        mel_min = mel_db.min()
        mel_max = mel_db.max()
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)

        mel_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)
        return mel_tensor
