from torchvision import transforms as T
import datetime
import os

DEVICE = "cuda"
DATA_DIR = r"D:\coding\python_projects\pytorch_projects\GAN\dcgan\face_dataset\celeb_dataset"
BATCH_SIZE = 64
IMG_SIZE = 128
ngf = 64
ndf = 64
Z_DIM = 100
nc = 3
lr = 2e-4
LAMBDA_GAN = 1.0
LAMBDA_FOURIER = 0.4
EPOCHS = 20
TENSORBOARD_FOLDERNAME = os.path.join(os.getcwd(), "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), "CelebA")
LOAD = False
MODEL_PATH = "model/"

transforms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

