import torch
from torch import nn
from config import DEVICE
import traceback


def compute_dft_torch(image_tensor: torch.Tensor) -> torch.Tensor:
    dft = torch.fft.fft2(image_tensor)
    dft_shift = torch.fft.fftshift(dft)
    magnitude_spec = torch.log(torch.abs(dft_shift) + 1)
    return magnitude_spec


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_checkpoint(checkpoint: dict, filename: str):
    try:
        torch.save(checkpoint, filename)
        print("Checkpoint saved successfully")
    except Exception as e:
        print(f"Checkpoint failed to save: {e}")


def load_checkpoint(checkpoint_file, generator, discriminator, optD, optG):
    try:
        checkpoints = torch.load(checkpoint_file, map_location=torch.device(DEVICE))
        
        generator.load_state_dict(checkpoints['generator'], strict=True)
        discriminator.load_state_dict(checkpoints['discriminator'], strict=True)
        optD.load_state_dict(checkpoints['opt_disc'])
        optG.load_state_dict(checkpoints['opt_gen'])

        print("Checkpoint loaded successfully")
    except Exception as e:
        print(f"Checkpoint loading failed: {e}")
        traceback.print_exc()