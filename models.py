import torch.nn as nn


class Conv1DAutoencoder(nn.Module):
    def __init__(
            self,
            len_sig: int,
            n_channels: int = 1
    ):
        super().__init__()

        # Encoder
        # Calculate the output size after the convolutional layers
        conv_output_size = 64 * (len_sig // 8)  # num_out_channels * time_steps_after_conv
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=n_channels, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_output_size, 2)  # Compress to 2 dimensions
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, conv_output_size),
            nn.ReLU(),
            nn.Unflatten(1, (64, len_sig // 8)),  # Reshape to (batch_size, 64, reduced_time_steps)
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=n_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Assuming the input signal is normalized
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
