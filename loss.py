import torch
from torch import nn
from config import DEVICE

class GanLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.gan_loss = nn.BCEWithLogitsLoss()

    def forward(self, real_preds, fake_preds):
        real_labels = torch.ones_like(real_preds).to(DEVICE)
        fake_labels = torch.zeros_like(fake_preds).to(DEVICE)

        real_gan_loss = self.gan_loss(real_preds, real_labels)
        fake_gan_loss = self.gan_loss(fake_preds, fake_labels)

        total_loss_gan =  (real_gan_loss + fake_gan_loss) / 2

        return total_loss_gan
    

class FourierLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.fourier_loss = nn.L1Loss()

    def forward(self,real_preds, fake_preds):
        real_labels = torch.ones_like(real_preds).to(DEVICE)
        fake_labels = torch.zeros_like(fake_preds).to(DEVICE)

        real_f_loss = self.fourier_loss(real_preds, real_labels)
        fake_f_loss = self.fourier_loss(fake_preds, fake_labels)

        total_fourier_loss = (real_f_loss + fake_f_loss) / 2

        return total_fourier_loss