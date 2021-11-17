import argparse
import os
import time
from glob import glob

import numpy as np
import pandas as pd
import torch.utils.data
import yaml

import unet.unet_helpers as utils
from unet import UNet

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
OUT_DIR = os.path.join(ROOT_DIR, 'out')
SEED = 0

utils.set_seed(SEED)

parser = argparse.ArgumentParser(description='PyTorch UNet Pose Evaluation')
parser.add_argument('data', metavar='TEST_DIR',
                    help='test dataset directory')
parser.add_argument('model', metavar='MODEL_NAME',
                    help='name of model in checkpoints directory')
parser.add_argument('aug_file', metavar='AUG_FILE',
                    help='path to augmentations file')
parser.add_argument('--channels', default=128, type=int, metavar='N',
                    help='number of channels in first convolutional layer (default: 128)')
args = parser.parse_args()

test_dir = args.data
checkpoint = os.path.join(CHECKPOINT_DIR, args.model, 'model_state.pt')
out_dir = os.path.join(OUT_DIR, args.model)
aug_file = os.path.abspath(args.aug_file)
channels = args.channels
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("test_dir:", test_dir)
print("checkpoint:", checkpoint)
print("out_dir:", out_dir)
print("aug_file:", aug_file)
print("device:", device)

os.makedirs(out_dir, exist_ok=True)
with open(aug_file, 'r') as f:
    aug_params = yaml.load(f, Loader=yaml.SafeLoader)

annotation_files = sorted(glob(os.path.join(test_dir, '*/*.h5')))
landmarks = pd.DataFrame()
for f in annotation_files:
    df = pd.read_hdf(f)
    df = df.iloc[np.argsort(df.T.columns)]
    landmarks = landmarks.append(df)
landmark_names = pd.unique(landmarks.columns.get_level_values('bodyparts'))

net = UNet(1, args.channels, landmark_names)
net.load_state_dict(torch.load(checkpoint))
net.to(device=device)

start_time = time.time()
if len(glob(os.path.join(test_dir, '*.png'))) == 0:
    folders = glob(os.path.join(test_dir, '*'))
else:
    folders = glob(os.path.join(test_dir))
for f in folders:
    prediction_df = utils.analyze_frames(f, landmark_names, aug_params['scorer'], net)
    prediction_df.to_hdf(os.path.join(out_dir, f.split('/')[-1] + '.h5'),
                         'df_with_missing',
                         format='table', mode='w')
print("Test frames prediction saved in", out_dir)
print("Inference Time:", time.time() - start_time)
