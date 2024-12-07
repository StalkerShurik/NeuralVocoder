import torch
from torch import nn
from torch.nn import Sequential
from torch.nn.utils.parametrizations import weight_norm


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: list,  # of M x L shape
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.activation = nn.LeakyReLU()

        self.m_len = len(dilations)
        self.l_len = len(dilations[0])

        self.conv = nn.ModuleList(
            nn.ModuleList(
                weight_norm(
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        dilation=dilations[m][li],
                        padding="same",
                    )
                )
                for li in range(self.l_len)
            )
            for m in range(self.m_len)
        )

    def forward(self, x):
        for m in range(self.m_len):
            x_res = x.clone()
            for li in range(self.l_len):
                x_res = self.activation(x_res)
                x_res = self.conv[m][li](x_res)
            x += x_res

        return x


class MRF(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: list,
        dilations: list,  # of N x M x L shape
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.n_len = len(dilations)

        self.ResBlocks = nn.ModuleList(
            ResBlock(channels, kernel_size[n], dilations[n]) for n in range(self.n_len)
        )

    def forward(self, x):
        out_x = torch.zeros_like(x)

        for n in range(self.n_len):
            out_x += self.ResBlocks[n](x)

        return out_x


class Generator(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        upsample_iters: int,
        upsample_kernels: list,
        MRF_params: dict,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        assert upsample_iters == len(upsample_kernels)

        self.upsample_iters = upsample_iters

        self.activation = nn.LeakyReLU()

        self.input_conv = weight_norm(
            nn.Conv1d(
                in_channels=80,
                out_channels=hidden_channels,
                kernel_size=7,
                dilation=1,
                padding="same",
            )
        )

        self.upsample = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    in_channels=hidden_channels // 2**i,
                    out_channels=hidden_channels // 2 ** (i + 1),
                    kernel_size=upsample_kernels[i],
                    stride=upsample_kernels[i] // 2,
                    padding=(upsample_kernels[i] - upsample_kernels[i] // 2) // 2,
                )
                for i in range(upsample_iters)
            ]
        )

        self.mrf = nn.ModuleList(
            MRF(**MRF_params, channels=hidden_channels // 2 ** (i + 1))
            for i in range(upsample_iters)
        )

        self.output_conv = nn.Sequential(
            nn.LeakyReLU(),
            weight_norm(
                nn.Conv1d(
                    in_channels=hidden_channels // 2 ** (upsample_iters),
                    out_channels=1,
                    kernel_size=7,
                    dilation=1,
                    padding="same",
                )
            ),
            nn.Tanh(),
        )

    def forward(self, input_spec, **kwargs):
        input_spec = input_spec[:, :, :100]  # GOD HELP

        x = self.input_conv(input_spec)

        for i in range(self.upsample_iters):
            x = self.activation(x)

            x = self.upsample[i](x)

            x = self.mrf[i](x)

        x = self.output_conv(x)

        return {"generated_wav": x.squeeze(1)}
