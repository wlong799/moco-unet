import argparse
import os
import time
from glob import glob

import numpy as np
import torch.nn as nn
import torch.utils.data
import yaml
from tqdm import tqdm

import unet.unet_helpers as utils
from unet import UNet

parser = argparse.ArgumentParser(description='PyTorch UNet Pose Estimation')
parser.add_argument('data', metavar='DATA_DIR', help='path to dataset')
parser.add_argument('aug_file', metavar='AUG_FILE', help='path to augmentations file')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--channels', default=128, type=int, metavar='N',
                    help='number of channels in first convolutional layer')

args = parser.parse_args()

t0 = time.time()

train_dir = os.path.join(args.data, 'train')
test_dir = os.path.join(args.data, 'test')
lr_init, batch_size, epochs, channels = args.lr, args.batch_size, args.epochs, args.channels
aug_file = args.aug_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Train dir:", train_dir)
print("Test dir:", test_dir)
print("aug_file:", aug_file)
cam = train_dir.split("/")[-2]
print("Train and test on images from:", cam)
print("device:", device)

model_name = "UNet_{c}_{lr}_{bs}_{ep}_{ch}".format(c=cam, lr=lr_init, bs=batch_size, ep=epochs, ch=channels)
print("model name:", model_name)
unet_dir = os.path.join(os.getcwd(), "checkpoints/" + model_name)
if not os.path.isdir(unet_dir):
    os.mkdir(unet_dir)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Get augmentation params ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
with open(aug_file, "r") as f:
    aug_params = yaml.load(f, Loader=yaml.SafeLoader)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dataset loader ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
train_dataset = utils.COCODataset(train_dir, multiview=False,
                                  img_xy=(aug_params['img_x'], aug_params['img_y']),
                                  scale=aug_params['unet_scale'],
                                  flip=aug_params['mirror'], rotation=aug_params['rotation'])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True)

#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Set network params ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# parameters related to training the network
learning_rate = lr_init  # initial learning rate
weight_decay = 1e-6  # 1e-8 # l2 regularization penalty #how learning rate decays over time
momentum = 0.9  # how much to use previous gradient direction
val_frac = 0.1  # what fraction of data to use for validation
seed = 0
utils.set_seed(seed)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Instantiate network ~~~~~~~~~~~~~~~~~~~~~~~~~~~
net = UNet(1, train_dataset.nlandmarks)
net.to(device=device)

# set learning rate schedule
lr_decay = 1.5
step_size = 10
LR = np.linspace(0, learning_rate, 10)
if epochs > 50:
    LR = np.append(LR, learning_rate * np.ones(20))  # nepochs-100))
    for i in range(epochs // step_size):
        LR = np.append(LR, LR[-1] / lr_decay * np.ones(step_size))
else:
    LR = np.append(LR, learning_rate * np.ones(max(0, epochs - 10)))

# gradient descent flavor
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
lrs = []
criterion = nn.BCEWithLogitsLoss()

# store loss per epoch
epoch_train_losses = np.zeros(epochs)
epoch_val_losses = []
epoch_train_losses[:] = np.nan

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train network ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Training network...")
t1 = time.time()
for epoch in range(epochs):
    train_loss = []
    optimizer.param_groups[0]['lr'] = LR[epoch]
    with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        # loop through each batch in the training data
        net.train()
        for batch in train_dataloader:
            # compute the loss
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            hm_labels = batch['heatmap'].to(device=device, dtype=torch.float32)
            hm_preds = net(imgs)  # evaluate network on batch
            loss = criterion(hm_preds, hm_labels)  # compute loss
            train_loss.append(loss.item())
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            # gradient descent
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.25)
            optimizer.step()
            pbar.update(imgs.shape[0])

    epoch_train_losses[epoch] = np.mean(train_loss)
    lrs.append(optimizer.param_groups[0]["lr"])

    if epoch % 10 == 0:
        tqdm.write('Train loss = %f' % np.mean(train_loss))

train_time = time.time() - t1
print("Network training complete")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Save model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
params = {"seed": seed,
          "batchsize": batch_size,
          "nepochs": epochs,
          "img_xy": (aug_params['img_x'], aug_params['img_y']),
          "LR": LR,
          "train_time": time.time() - t1}
torch.save({'model': net,
            'optimizer_state_dict': optimizer.state_dict(),
            "train_loss": epoch_train_losses,
            'params': params},
           os.path.join(unet_dir, f'model_params.pth'))
torch.save(net.state_dict(), os.path.join(unet_dir, f'model_state.pt'))

print("Model training completed! Model saved in", unet_dir)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluate test frames ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Analyzing test frames")
t2 = time.time()
bodyparts = train_dataset.landmark_names
if len(glob(os.path.join(test_dir, "*.png"))) == 0:
    folders = glob(os.path.join(test_dir, "*"))  # subfolders present
else:
    folders = glob(os.path.join(test_dir))
for f in folders:
    pred_df = utils.analyze_frames(f, bodyparts, aug_params['scorer'], net,
                                   (aug_params['img_x'], aug_params['img_y']))

    pred_df.to_hdf(os.path.join(unet_dir, f.split("/")[-1] + "_" + model_name + ".h5"), "df_with_missing",
                   format="table", mode="w")
inference_time = time.time() - t2
print("Test frames prediction saved in", unet_dir)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Time summary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("~~~~~~~~~~~~~~ Time summary ~~~~~~~~~~~~~~~~~")
print("Training:", train_time)
print("Inference:", inference_time)
print("Run time:", time.time() - t0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ### End ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~