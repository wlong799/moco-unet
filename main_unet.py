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
parser.add_argument('data', metavar='DATA_DIR',
                    help='path to dataset')
parser.add_argument('checkpoints', metavar='SAVE_DIR',
                    help='where to save checkpoints')
parser.add_argument('aug_file', metavar='AUG_FILE',
                    help='path to augmentations file')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='total mini-batch size across GPUs (default: 256)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float, metavar='LR', dest='lr',
                    help='initial learning rate (default: 0.03)')
parser.add_argument('--channels', default=128, type=int, metavar='N',
                    help='number of channels in first convolutional layer (default: 128)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to moco pretrained checkpoint')

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

unet_dir = args.checkpoints
os.makedirs(unet_dir, exist_ok=True)

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

# load from pre-trained moco file
if args.pretrained:
    if os.path.isfile(args.pretrained):
        print("loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        args.start_epoch = 0
        msg = net.load_state_dict(state_dict, strict=False)
        # verify that missing data is only from second half of model
        for key in msg.missing_keys:
            if key.startswith('up') or key.startswith('outc'):
                continue
            raise Exception(f"'{key}' was not found in pretrained model")
        print("loaded pre-trained model '{}'".format(args.pretrained))
    else:
        print("no checkpoint found at '{}'".format(args.pretrained))

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
parameters = list(filter(lambda p: p.requires_grad, net.parameters()))
optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
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

    pred_df.to_hdf(os.path.join(unet_dir, f.split("/")[-1] + ".h5"),
                   "df_with_missing",
                   format="table", mode="w")
inference_time = time.time() - t2
print("Test frames prediction saved in", unet_dir)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Time summary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("~~~~~~~~~~~~~~ Time summary ~~~~~~~~~~~~~~~~~")
print("Training:", train_time)
print("Inference:", inference_time)
print("Run time:", time.time() - t0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ### End ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
