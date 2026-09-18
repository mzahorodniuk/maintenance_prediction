import torch
import torch.nn as nn

class DAE(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(208 * 3, 256),
            nn.ReLU(),

            nn.Linear(256, 32),
            nn.ReLU(),

            nn.Linear(32, 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),

            nn.Linear(32, 256),
            nn.ReLU(),

            nn.Linear(256, 208 * 3),
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
    
class DAEParam(nn.Module):
    def __init__(self, inout):
        super(DAEParam, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(inout, 32),
            nn.ReLU(),

            nn.Linear(32, 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),

            nn.Linear(32, inout),
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
    
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=2, padding=1), # 3, 208 -> 16, 104   (C, L)
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1), # 16, 104 -> 32, 52
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1), # 32, 52 -> 64, 26
            nn.ReLU(),
            nn.Conv1d(64, 8, kernel_size=3, stride=2, padding=1), # 64, 26 -> 8, 13   ## 8/20
            #nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
    
# Latent space smaller? / fc last layer in encoder then latent?
# FC layers for decoder?
class CAETest(nn.Module):
    def __init__(self):
        super(CAETest, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 3, kernel_size=3, stride=2, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(3, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
    
class CAEDenseDecoder(nn.Module):
    def __init__(self):
        super(CAEDenseDecoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, stride=2, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1 * 52, 16 * 104),
            nn.ReLU(),
            nn.Linear(16 * 104, 3 * 208),
            nn.Unflatten(1, (3, 208)),
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded