"""
Convolutional autoencoder
https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html
"""
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, c_hid=16, latent_dim=100):
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, c_hid, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 16 * c_hid, latent_dim),
        )
    def forward(self, x):
        return self.network(x)
    
class Decoder(nn.Module):
    def __init__(self, c_hid=16, latent_dim=100):
        super(Decoder, self).__init__()
        self.linear = nn.Sequential(nn.Linear(latent_dim, 2 * 16 * c_hid), nn.ReLU())
        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                c_hid, 3, kernel_size=3, output_padding=1, padding=1, stride=2
            ),
            nn.Tanh(),
        )
    def forward(self, x):
        z = self.linear(x)
        z = z.reshape(z.shape[0], -1, 4, 4)
        return self.network(z)

class AutoEncoder(nn.Module):
    def __init__(self, c_hid=16, latent_dim=100):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(c_hid, latent_dim)
        self.decoder = Decoder(c_hid, latent_dim)
    def forward(self, x):
        return self.decoder(self.encoder(x))
    def get_features(self, x):
        return self.encoder(x)