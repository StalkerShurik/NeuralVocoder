import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm, weight_norm


class SubDiscriminatorMPD(nn.Module):
    def __init__(self, period: int, repeats: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.period = period
        self.repeats = repeats

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv2d(
                        in_channels=2 ** (i if i == 0 else i + 4),
                        out_channels=2 ** (i + 5),
                        kernel_size=(5, 1),
                        stride=(3, 1),
                        padding=(2, 0),
                    )
                )
                for i in range(self.repeats)
            ]
        )
        self.activation = nn.LeakyReLU()

        self.mid_conv = weight_norm(
            nn.Conv2d(
                in_channels=2 ** (self.repeats + 4),
                out_channels=2 ** (self.repeats + 4),
                kernel_size=(5, 1),
                padding=(2, 0),
            )
        )

        self.output_conv = weight_norm(
            nn.Conv2d(
                in_channels=2 ** (self.repeats + 4),
                out_channels=1,
                kernel_size=(3, 1),
                padding=(1, 0),
            )
        )

    def forward(self, x):
        batch_size, time = x.shape

        if time % self.period != 0:
            x = F.pad(x, (0, self.period - time % self.period))
            batch_size, time = x.shape

        x = x.reshape(batch_size, 1, time // self.period, self.period)

        hidden_features = []

        for i in range(self.repeats):
            x = self.convs[i](x)
            x = self.activation(x)
            hidden_features.append(x)
        x = self.mid_conv(x)
        x = self.activation(x)
        x = self.output_conv(x)
        return torch.flatten(x, start_dim=1, end_dim=-1), hidden_features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(
        self, periods: list = [2, 3, 5, 7, 11], sub_repeats=4, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.sub_discriminators = nn.ModuleList(
            SubDiscriminatorMPD(period=periods[i], repeats=sub_repeats)
            for i in range(len(periods))
        )

    def forward(self, real_wav, generated_wav):
        hidden_features1 = []
        hidden_features2 = []

        real_outputs = []
        generated_outputs = []

        for disriminator in self.sub_discriminators:
            wav_real_processed, hidden_features1_sample = disriminator(real_wav)
            wav_generated_processed, hidden_features2_sample = disriminator(
                generated_wav
            )

            hidden_features1.append(hidden_features1_sample)
            hidden_features2.append(hidden_features2_sample)

            real_outputs.append(wav_real_processed)
            generated_outputs.append(wav_generated_processed)

        return hidden_features1, real_outputs, hidden_features2, generated_outputs


class SubDiscriminatorMSD(nn.Module):
    def __init__(self, is_first_block=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        parametrization_type = spectral_norm if is_first_block else weight_norm

        self.activation = nn.LeakyReLU()

        self.convs = nn.ModuleList(
            [
                parametrization_type(
                    nn.Conv1d(
                        in_channels=1,
                        out_channels=16,
                        kernel_size=15,
                        stride=1,
                        padding=7,
                    )
                ),
                parametrization_type(
                    nn.Conv1d(
                        in_channels=16,
                        out_channels=64,
                        kernel_size=41,
                        stride=4,
                        groups=4,
                        padding=20,
                    )
                ),
                parametrization_type(
                    nn.Conv1d(
                        in_channels=64,
                        out_channels=256,
                        kernel_size=41,
                        stride=4,
                        groups=16,
                        padding=20,
                    )
                ),
                parametrization_type(
                    nn.Conv1d(
                        in_channels=256,
                        out_channels=1024,
                        kernel_size=41,
                        stride=4,
                        groups=64,
                        padding=20,
                    )
                ),
                parametrization_type(
                    nn.Conv1d(
                        in_channels=1024,
                        out_channels=1024,
                        kernel_size=41,
                        stride=4,
                        groups=256,
                        padding=20,
                    )
                ),
                parametrization_type(
                    nn.Conv1d(
                        in_channels=1024,
                        out_channels=1024,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                    )
                ),
            ]
        )

        self.output_conv = parametrization_type(
            nn.Conv1d(
                in_channels=1024, out_channels=1, kernel_size=3, stride=1, padding=1
            )
        )

    def forward(self, x):
        hidden_features = []

        x = x.unsqueeze(1)

        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = self.activation(x)
            hidden_features.append(x)

        x = self.output_conv(x)

        return x, hidden_features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers = nn.ModuleList(
            [
                SubDiscriminatorMSD(is_first_block=True),
                SubDiscriminatorMSD(),
                SubDiscriminatorMSD(),
            ]
        )
        self.pool = nn.ModuleList(
            [nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)]
        )

    def forward(self, real_wav, generated_wav):
        hidden_features1 = []
        hidden_features2 = []

        real_outputs = []
        generated_outputs = []

        for i, disriminator in enumerate(self.layers):
            if i > 0:
                real_wav = self.pool[i - 1](real_wav)
                generated_wav = self.pool[i - 1](generated_wav)

            wav_real_processed, hidden_features1_sample = disriminator(real_wav)
            wav_generated_processed, hidden_features2_sample = disriminator(
                generated_wav
            )

            hidden_features1.append(hidden_features1_sample)
            hidden_features2.append(hidden_features2_sample)

            real_outputs.append(wav_real_processed)
            generated_outputs.append(wav_generated_processed)

        return hidden_features1, real_outputs, hidden_features2, generated_outputs
