import torch
from torch import nn

from src.transforms.get_spec import MelSpectrogram, MelSpectrogramConfig


class DiscriminatorLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, real: list, fake: list):
        loss = 0

        for sample_real, sample_fake in zip(real, fake):
            loss += torch.mean(sample_real - 1) ** 2 + torch.mean(sample_fake) ** 2

        return loss


class FeatureMatchingLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, hidden_features1, hidden_features2):
        loss = 0
        for sub1, sub2 in zip(hidden_features1, hidden_features2):
            for hf1, hf2 in zip(sub1, sub2):
                loss += torch.mean(torch.abs(hf1 - hf2))

        return 2 * loss  # lambda = 2


class MelLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, wav1, wav2):
        mel_spec = MelSpectrogram(MelSpectrogramConfig())

        mel_wav1 = mel_spec(wav1)
        mel_wav2 = mel_spec(wav2)

        return 45 * torch.abs(mel_wav1 - mel_wav2).sum()


class GANLossGenerator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, fake: list):
        loss = 0

        for sample_fake in fake:
            loss += torch.mean(sample_fake - 1) ** 2

        return loss


class GeneratorLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.feature_matching = FeatureMatchingLoss()
        self.mel_loss = MelLoss()
        self.gan_loss = GANLossGenerator()

    def forward(self, hidden_features1, hidden_features2, wav1, wav2, fake):
        loss = self.feature_matching(hidden_features1, hidden_features2)
        loss += self.mel_loss(wav1, wav2)
        loss += self.gan_loss(fake)

        return loss
