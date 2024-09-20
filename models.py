import torch
import torch.nn as nn


class Conv1DAutoencoder(nn.Module):
    def __init__(
            self,
            len_sig: int,
            n_channels: int = 1,
            n_depth: int = 3,
            n_channels_out_0: int = 16,
            stride: int = 2,
            device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        # Calculate the output size after the convolutional layers
        n_channels_out_max = n_channels_out_0 * 2**(n_depth - 1)
        size_factor = len_sig // (stride ** n_depth)
        conv_output_size = n_channels_out_max * size_factor

        # Autoencoder architecture:
        # The encoder consists of <n_depth> blocks of 1D convolutions followed by ReLU activation functions.
        # They are followed by a flattening operation and a linear (dense) layer with only  two output neurons
        # (as the signal is compressed to two dimensions.)
        # The decoder start from these two neurons, enlarging them by a linear (dense) layer and up-scaling from
        # this layer with

        # --- Encoder
        layers = []
        # initialize variables
        in_channels = n_channels
        out_channels = n_channels_out_0
        for i in range(n_depth):
            # add one block
            layers += [
                # convolution layer
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    device=device
                ),
                # activation function
                nn.ReLU()
            ]
            # update variables
            in_channels = out_channels
            out_channels *= 2

        layers += [
            # flatten so that the input to the dense layer is an array
            nn.Flatten(),
            # dense layer
            nn.Linear(conv_output_size, 2)  # Compress to 2 dimensions
        ]
        # build model
        self.encoder = nn.Sequential(*layers)

        # --- Decoder
        layers = [
            # dense layer that enlarges the two-neuron-signal to higher dimensions
            nn.Linear(2, conv_output_size),
            nn.ReLU(),
            # unflatten the input again to the minimal shape of the signal
            nn.Unflatten(1, (n_channels_out_max, size_factor)),  # Reshape to (batch_size, 64, reduced_time_steps)
        ]

        for i in range(n_depth):
            if i >= (n_depth - 1):
                # the last layer requires special treatment: shape of the output signals and a Sigmoid activation
                # function assuming a normalized signal [-1, +1]
                activation_fnc = nn.Sigmoid()
                out_channels = n_channels
            else:
                # all other (previous) layers have a ReLU activation function
                activation_fnc = nn.ReLU()
                out_channels //= 2

            layers += [
                nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    output_padding=1,
                    device=device
                ),
                activation_fnc,
            ]
            # update variables
            in_channels = out_channels

        # build model
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
