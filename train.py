import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from loss import GanLoss, FourierLoss
from config import *
from model import *
from utils import *


train_dataset = ImageFolder(
    root=DATA_DIR,
    transform=transforms
)

dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

netG = Generator128(Z_DIM, nc, ngf).to(DEVICE)
netD = Discriminator128(nc, ndf).to(DEVICE)
netG.apply(weights_init)
netD.apply(weights_init)

fixed_noise = torch.rand((BATCH_SIZE, Z_DIM, 1,1)).to(DEVICE)

optimizer_G = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))


# if LOAD:
#     load_checkpoint(MODEL_PATH + "gan_fft_128_1.pt", netG, netD, optimizer_D, optimizer_G)


gan_loss_fn = GanLoss().to(DEVICE)
fourier_loss_fn = FourierLoss().to(DEVICE)

writer_real = SummaryWriter(f"{TENSORBOARD_FOLDERNAME}/real")
writer_fake = SummaryWriter(f"{TENSORBOARD_FOLDERNAME}/fake")
disc_graph = SummaryWriter(f"{TENSORBOARD_FOLDERNAME}/disc_graph")
gen_graph = SummaryWriter(f"{TENSORBOARD_FOLDERNAME}/gen_graph")
step = 0



for epoch in range(EPOCHS):

    loader = tqdm(dataloader, leave=False, desc=f"Epochs[{epoch}/{EPOCHS}]")

    for idx, (real_img, _) in enumerate(loader):
        real_img = real_img.to(DEVICE)
        cur_batch_size = real_img.shape[0]
        
        noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(DEVICE)
        fake_img = netG(noise)
        
        real_pred = netD(real_img)
        fake_pred_detached = netD(fake_img.detach())
        
        ganloss = gan_loss_fn(real_pred, fake_pred_detached)
        
        real_fft = compute_dft_torch(real_img)
        fake_fft = compute_dft_torch(fake_img)

        real_fft_pred = netD(real_fft)
        fake_fft_pred = netD(fake_fft)

        fourierloss = fourier_loss_fn(real_fft_pred, fake_fft_pred)
        
        total_disc_loss = (ganloss * LAMBDA_GAN) + (fourierloss * LAMBDA_FOURIER)
        
        optimizer_D.zero_grad()
        total_disc_loss.backward(retain_graph=True)
        optimizer_D.step()

        fake_pred = netD(fake_img)
        
        gen_loss = gan_loss_fn(fake_pred, torch.ones_like(fake_pred))
        
        optimizer_G.zero_grad()
        gen_loss.backward()
        optimizer_G.step()

        if idx % 10 == 0 and idx > 0:
            print(
                f"Epoch [{epoch}/{EPOCHS}] Batch {idx}/{len(loader)} "
                f"Loss D: {total_disc_loss:.4f}, Loss G: {gen_loss:.4f}"
            )

            with torch.no_grad():
                fake = netG(fixed_noise)
                
                img_grid_real = torchvision.utils.make_grid(real_img[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                disc_graph.add_scalar("disc loss", total_disc_loss, global_step=step)
                gen_graph.add_scalar("Gen Loss", gen_loss, global_step=step)

            step += 1

    

    checkpoint = {
        "generator": netG.state_dict(),
        "discriminator": netD.state_dict(),
        "opt_gen": optimizer_G.state_dict(),
        "opt_disc": optimizer_D.state_dict(),
        "epochs": epoch
    }

    save_checkpoint(checkpoint, f"model/gan_fft_128_{epoch}.pt")