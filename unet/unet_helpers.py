import itertools
import os
import random
from glob import glob

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def heatmap2landmarks_torch(hms):
    idx = torch.argmax(hms.reshape(hms.shape[:-2] + (hms.shape[-2] * hms.shape[-1],)), axis=-1)
    coord = unravel_index(idx, hms.shape[-2:])
    return coord


def unravel_indices(indices, shape):
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord


def unravel_index(indices, shape):
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (N,).
        shape: The targeted shape, (D,).

    Returns:
        A tuple of unraveled coordinate tensors of shape (D,).
    """
    coord = unravel_indices(indices, shape)
    return coord


# DeeperCut's methods

## Local refinement
def set_locref(locref_map, locref_mask, locref_scale, j, i, j_id, dx, dy):
    locref_mask[j, i, j_id * 2 + 0] = 1
    locref_mask[j, i, j_id * 2 + 1] = 1
    locref_map[j, i, j_id * 2 + 0] = dx * locref_scale
    locref_map[j, i, j_id * 2 + 1] = dy * locref_scale


def compute_locref_maps(landmarks, size, landmark_names, lowres=False):
    # Set params
    stride = 4
    half_stride = stride / 2
    scale = 1

    if lowres:
        size = np.ceil(np.array(size) / stride).astype(int)  # (stride * 2)).astype(int) * 2
        dist_thresh = 4 * scale
        locref_stdev = 8
    else:
        dist_thresh = 10 * scale
        locref_stdev = 17.2801
    width = size[1]
    height = size[0]
    dist_thresh_sq = dist_thresh ** 2

    nlandmarks = len(landmark_names)
    scmap = np.zeros(np.concatenate([size, np.array([nlandmarks])]))
    locref_shape = np.concatenate([size, np.array([nlandmarks * 2])])
    locref_mask = np.zeros(locref_shape)
    locref_map = np.zeros(locref_shape)

    for k, j_id in enumerate(landmark_names):
        joint_pt = landmarks[k][::-1]
        j_x = joint_pt[0]
        j_y = joint_pt[1]
        if ~np.isnan(j_x) or ~np.isnan(j_y):
            # don't loop over entire heatmap, but just relevant locations
            if lowres:
                j_x_sm = round((j_x - half_stride) / stride)
                j_y_sm = round((j_y - half_stride) / stride)
            else:
                j_x_sm = (j_x)
                j_y_sm = (j_y)
            min_x = round(max(j_x_sm - dist_thresh - 1, 0))
            max_x = round(min(j_x_sm + dist_thresh + 1, width - 1))
            min_y = round(max(j_y_sm - dist_thresh - 1, 0))
            max_y = round(min(j_y_sm + dist_thresh + 1, height - 1))

            for j in range(min_y, max_y + 1):  # range(height):
                if lowres:
                    pt_y = j * stride + half_stride
                else:
                    pt_y = j
                for i in range(min_x, max_x + 1):  # range(width):
                    if lowres:
                        pt_x = i * stride + half_stride
                    else:
                        pt_x = i
                    dx = j_x - pt_x
                    dy = j_y - pt_y
                    dist = dx ** 2 + dy ** 2
                    if dist <= dist_thresh_sq:
                        dist = dx ** 2 + dy ** 2
                    locref_scale = 1.0 / locref_stdev
                    set_locref(locref_map, locref_mask, locref_scale, j, i, k, dx, dy)

    return locref_map.T, locref_mask.T


def argmax_pose_predict(scmap, offmat):
    """Combine scoremat and offsets to the final pose."""
    num_joints = scmap.shape[0]
    pose = []
    landmarks = heatmap2landmarks(scmap)
    for joint_idx in range(num_joints):
        maxloc = landmarks[joint_idx].astype(int)
        offset_x = offmat[joint_idx * 2][maxloc[0]][maxloc[1]]
        offset_y = offmat[joint_idx * 2 + 1][maxloc[0]][maxloc[1]]
        maxloc_w_offset = (maxloc[0] + offset_x, maxloc[1] + offset_y)
        pose.append(maxloc_w_offset)
    return np.array(pose)


def heatmap2landmarks(hms):
    idx = np.argmax(hms.reshape(hms.shape[:-2] + (hms.shape[-2] * hms.shape[-1],)), axis=-1)
    locs = np.zeros(hms.shape[:-2] + (2,))
    locs[..., 1], locs[..., 0] = np.unravel_index(idx, hms.shape[-2:])
    return locs


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def PlotLabelAndCoordsPrediction(batch, coords, ncolors, idx=None, title_string=''):
    """
    PlotLabelAndCoordsPrediction(batch,pred,idx=None):
    Plot the input, labels, and label predictions for a batch.
    """
    cmap = matplotlib.cm.get_cmap('jet')
    colornorm = matplotlib.colors.Normalize(vmin=0, vmax=ncolors)
    colors = cmap(colornorm(np.arange(ncolors)))
    isbatch = isinstance(batch['id'], torch.Tensor)

    if idx is None and isbatch:
        idx = range(len(batch['id']))
    if isbatch:
        n = len(idx)
    else:
        n = 1
        idx = [None, ]
    locs_pred = coords_pred  # .cpu().numpy()

    for i in range(n):
        plt.subplot(n, 4, 4 * i + 1)
        im = COCODataset.get_image(batch, idx[i])
        plt.imshow(im, cmap='gray')
        locs = COCODataset.get_landmarks(batch, idx[i])
        for k in range(train_dataset.nlandmarks):
            plt.plot(locs[k, 0], locs[k, 1], marker='.', color=colors[k], markerfacecolor=colors[k])
        if isbatch:
            batchid = batch['id'][i]
        else:
            batchid = batch['id']
        plt.title(title_string + '%d' % batchid)

        plt.subplot(n, 4, 4 * i + 2)
        plt.imshow(im, cmap='gray')
        for k in range(train_dataset.nlandmarks):
            plt.scatter(locs_pred[i, k, 0], locs_pred[i, k, 1], color=colors[k])
        if i == 0: plt.title('pred')

    # helper function for plotting outputs


def PlotLabelAndPrediction(batch, hm_pred, ncolors, idx=None, title_string=''):
    """
    PlotLabelAndPrediction(batch,pred,idx=None):
    Plot the input, labels, and predictions for a batch.
    """
    cmap = matplotlib.cm.get_cmap('jet')
    colornorm = matplotlib.colors.Normalize(vmin=0, vmax=ncolors)
    colors = cmap(colornorm(np.arange(ncolors)))
    isbatch = isinstance(batch['id'], torch.Tensor)

    if idx is None and isbatch:
        idx = range(len(batch['id']))
    if isbatch:
        n = len(idx)
    else:
        n = 1
        idx = [None, ]
    locs_pred = heatmap2landmarks(hm_pred.cpu().numpy())
    for i in range(n):

        plt.subplot(n, 4, 4 * i + 1)
        im = COCODataset.get_image(batch, idx[i])
        plt.imshow(im, cmap='gray')
        locs = COCODataset.get_landmarks(batch, idx[i])
        for k in range(train_dataset.nlandmarks):
            plt.plot(locs[k, 0], locs[k, 1], marker='.', color=colors[k], markerfacecolor=colors[k])
        if isbatch:
            batchid = batch['id'][i]
        else:
            batchid = batch['id']
        plt.title(title_string + '%d' % batchid)

        plt.subplot(n, 4, 4 * i + 2)
        plt.imshow(im, cmap='gray')
        locs = COCODataset.get_landmarks(batch, idx[i])
        if isbatch:
            locs_pred_curr = locs_pred[i, ...]
        else:
            locs_pred_curr = locs_pred
        for k in range(train_dataset.nlandmarks):
            plt.plot(locs_pred_curr[k, 0], locs_pred_curr[k, 1], marker='.', color=colors[k], markerfacecolor=colors[k])
        if i == 0: plt.title('pred')

        plt.subplot(n, 4, 4 * i + 3)
        hmim = COCODataset.get_heatmap_image(batch, idx[i])
        plt.imshow(hmim)
        if i == 0: plt.title('label')

        plt.subplot(n, 4, 4 * i + 4)
        if isbatch:
            predcurr = hm_pred[idx[i], ...]
        else:
            predcurr = hm_pred
        plt.imshow(heatmap2image(predcurr.cpu().numpy(), colors=colors))
        if i == 0: plt.title('pred')


def analyze_frames(frames_dir, bodyparts, scorer, net, img_xy):
    """
    Input:-
    - frames_dir: path containing .png files
    - bodyparts: landmark names
    - scorer: Name of scorer
    - net: UNet model for predictions
    - img_xy: size of images for resizing before predictions
    Output:-
    - dataFrame: predictions from network stored as multi-level dataframe
    """
    frames = glob(frames_dir + "/*.png")
    # Create an empty dataframe
    for index, bodypart in enumerate(bodyparts):
        columnindex = pd.MultiIndex.from_product(
            [[scorer], [bodypart], ["x", "y"]],
            names=["scorer", "bodyparts", "coords"])
        frame = pd.DataFrame(
            np.nan,
            columns=columnindex,
            index=[os.path.join(fn.split("/")[-2], fn.split("/")[-1]) for fn in frames])
        if index == 0:
            dataFrame = frame
        else:
            dataFrame = pd.concat([dataFrame, frame], axis=1)
    # Add predicted values to dataframe
    net.eval()
    for ind, img_file in enumerate(frames):
        im = cv2.imread(img_file, 0)
        if im.dtype == float:
            pass
        elif im.dtype == np.uint8:
            im = im.astype(float) / 255.
        elif im.dtype == np.uint16:
            im = im.astype(float) / 65535.
        else:
            print('Cannot handle im type ' + str(im.dtype))
            raise TypeError
        # im = resize(im, img_xy, anti_aliasing=False)
        im = torch.Tensor(im[np.newaxis, np.newaxis, :, :])
        pred = torch.sigmoid(net(im.to(device=device, dtype=torch.float32)))
        landmarks = heatmap2landmarks(pred.cpu().detach().numpy()).ravel()
        dataFrame.iloc[ind] = landmarks
    return dataFrame


def heatmap2image(hm, cmap='jet', colors=None):
    """
    heatmap2image(hm,cmap='jet',colors=None)
    Creates and returns an image visualization from landmark heatmaps. Each
    landmark is colored according to the input cmap/colors.
    Inputs:
    hm: nlandmarks x height x width ndarray, dtype=float in the range 0 to 1.
    hm[p,i,j] is a score indicating how likely it is that the pth landmark
    is at pixel location (i,j).
    cmap: string.
    Name of colormap for defining colors of landmark points. Used only if colors
    is None.
    Default: 'jet'
    colors: list of length nlandmarks.
    colors[p] is an ndarray of size (4,) indicating the color to use for the
    pth landmark. colors is the output of matplotlib's colormap functions.
    Default: None
    Output:
    im: height x width x 3 ndarray
    Image representation of the input heatmap landmarks.
    """
    hm = np.maximum(0., np.minimum(1., hm))
    im = np.zeros((hm.shape[1], hm.shape[2], 3))
    if colors is None:
        if isinstance(cmap, str):
            cmap = matplotlib.cm.get_cmap(cmap)
        colornorm = matplotlib.colors.Normalize(vmin=0, vmax=hm.shape[0])
        colors = cmap(colornorm(np.arange(hm.shape[0])))
    for i in range(hm.shape[0]):
        color = colors[i]
        for c in range(3):
            im[..., c] = im[..., c] + (color[c] * .7 + .3) * hm[i, ...]
    im = np.minimum(1., im)
    return im


def image_augmentation(X, xy, motion_blur=True):
    """
    X: LIST of ND-arrays, float
        list of image arrays of size [nchan x Ly x Lx] or [Ly x Lx]
    """
    nimg = len(X)
    if X[0].ndim > 2:
        nchan = X[0].shape[0]
    else:
        nchan = 1
    imgi = np.zeros((nimg, nchan, xy[0], xy[1]), np.float32)
    # Specify the kernel size.
    # The greater the size, the more the motion.
    kernel_size = 20
    # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))
    # Create a copy of the same for creating the horizontal kernel.
    kernel_h = np.copy(kernel_v)
    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

    # Normalize.
    kernel_v /= kernel_size
    kernel_h /= kernel_size
    for n in range(nimg):
        Ly, Lx = X[n].shape[-2:]
        imgi[n] = X[n].copy()

        if motion_blur:
            # Apply the vertical kernel.
            imgi[n] = cv2.filter2D(imgi[n], -1, kernel_v)
            # Apply the horizontal kernel.
            imgi[n] = cv2.filter2D(imgi[n], -1, kernel_h)
    return imgi


def random_rotate_and_resize(X, Y, scale_range=1., xy=(300, 300), do_flip=True, rotation=10):
    """ augmentation by random rotation and resizing
        X and Y are lists or arrays of length nimg, with dims channels x Ly x Lx (channels optional)
        Parameters
        ----------
        X: LIST of ND-arrays, float
            list of image arrays of size [nchan x Ly x Lx] or [Ly x Lx]
        Y: LIST of ND-arrays, float (optional, default None)
            list of image labels of size [nlabels x Ly x Lx] or [Ly x Lx]. The 1st channel
            of Y is always nearest-neighbor interpolated (assumed to be masks or 0-1 representation).
            If Y.shape[0]==3 and not unet, then the labels are assumed to be [cell probability, Y flow, X flow].
            If unet, second channel is dist_to_bound.
        scale_range: float (optional, default 1.0)
            Range of resizing of images for augmentation. Images are resized by
            (1-scale_range/2) + scale_range * np.random.rand()
        xy: tuple, int (optional, default (224,224))
            size of transformed images to return
        do_flip: bool (optional, default True)
            whether or not to flip images horizontally
        rescale: array, float (optional, default None)
            how much to resize images by before performing augmentations
        unet: bool (optional, default False)
        Returns
        -------
        imgi: ND-array, float
            transformed images in array [nimg x nchan x xy[0] x xy[1]]
        lbl: ND-array, float
            transformed labels in array [nimg x nchan x xy[0] x xy[1]]
        scale: array, float
            amount each image was resized by
    """
    scale_range = max(0, min(2, float(scale_range)))
    nimg = len(X)
    if X[0].ndim > 2:
        nchan = X[0].shape[0]
    else:
        nchan = 1
    imgi = np.zeros((nimg, nchan, xy[0], xy[1]), np.float32)

    lbl = []
    if Y[0].ndim > 2:
        nt = Y[0].shape[0]
    else:
        nt = 1
    lbl = np.zeros((nimg, nt, xy[0], xy[1]), np.float32)
    scale = np.zeros(nimg, np.float32)

    for n in range(nimg):
        Ly, Lx = X[n].shape[-2:]
        imgi[n], lbl[n] = X[n].copy(), Y[n].copy()

        # generate random augmentation parameters
        flip = np.random.rand() > .5
        theta = np.random.randint(rotation)  # random degree of rotation #np.random.rand() * np.pi * 2
        scale[n] = (1 - scale_range / 2) + scale_range * np.random.rand()

        # create affine transform
        c = (xy[0] * 0.5 - 0.5, xy[1] * 0.5 - 0.5)  # Center of image
        M = cv2.getRotationMatrix2D(c, theta, scale[n])

        if flip and do_flip:
            imgi[n] = np.flip(imgi[n], axis=-1)
            lbl[n] = np.flip(lbl[n], axis=-1)

        for k in range(nchan):
            I = cv2.warpAffine(imgi[n][k], M, (xy[1], xy[0]), flags=cv2.INTER_LINEAR)
            imgi[n] = I

        for k in range(nt):
            lbl[n, k] = cv2.warpAffine(lbl[n][k], M, (xy[1], xy[0]), flags=cv2.INTER_LINEAR)

    # imgi = image_augmentation(imgi, xy) # motion blur
    return imgi, lbl, scale


class COCODataset(torch.utils.data.Dataset):
    """
    COCODataset
    Torch Dataset based on the COCO keypoint file format.
    """

    def __init__(self, datadir, annfile=None, multiview=False, img_xy=(300, 300),
                 scale=0.5, flip=True, rotation=20):

        self.label_filter = None
        self.label_filter_r = 5  # 5
        self.label_filter_d = 10
        self.label_sigma = 8
        self.init_label_filter()
        self.flip = flip
        self.img_xy = img_xy
        self.rotation = rotation
        self.scale = scale

        if multiview:
            print("processing multiview files")
            views = glob(os.path.join(datadir, "*"))
            self.img_files = []
            for v in views:
                view = v.split("/")[-1]
                self.img_files.append(sorted(glob(os.path.join(datadir, "{}/*/*.png".format(view)))))
                annfiles = sorted(glob(os.path.join(datadir, "{}/*/*.h5".format(view))))
                self.img_files = list(itertools.chain(*self.img_files))
        else:
            print("processing single view files")
            self.datadir = datadir
            self.img_files = sorted(glob(os.path.join(self.datadir, '*/*.png')))
            # Lanmarks/key points info
            annfiles = sorted(glob(os.path.join(self.datadir, '*/*.h5')))

        # Landmarks dataframe concatentation
        self.landmarks = pd.DataFrame()
        for f in annfiles:
            df = pd.read_hdf(f)
            df = df.iloc[np.argsort(df.T.columns)]  # sort annotations to match img_file order
            self.landmarks = self.landmarks.append(df)

        self.im = []
        for file in self.img_files:
            im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, img_xy, cv2.INTER_AREA)
            # convert to float32 in the range 0. to 1.
            if im.dtype == float:
                pass
            elif im.dtype == np.uint8:
                im = im.astype(float) / 255.
            elif im.dtype == np.uint16:
                im = im.astype(float) / 65535.
            else:
                print('Cannot handle im type ' + str(im.dtype))
                raise TypeError
            # im = self.normalize99(im)  # Normalize images
            if im.ndim < 3:
                self.im.append(im[np.newaxis, ...])

        self.landmark_names = pd.unique(self.landmarks.columns.get_level_values("bodyparts"))
        self.nlandmarks = len(self.landmark_names)

        # Create heatmaps for target prediction and location refinement maps
        self.landmark_heatmaps = []
        self.locref_map, self.locref_mask = [], []  # locref map and mask
        self.lowres_locref_map = []
        for i in range(len(self.im)):
            # locs: y_pos x x_pos for data augmentation
            locs = np.array([self.landmarks.values[i][::2], self.landmarks.values[i][1::2]]).T
            target = self.make_heatmap_target(locs, np.squeeze(self.im[i]).shape)
            self.landmark_heatmaps.append(target.detach().numpy())
            locref_map, _ = compute_locref_maps(locs, self.img_xy, self.landmark_names)
            lowres_locref, _ = compute_locref_maps(locs, self.img_xy,
                                                   self.landmark_names, lowres=True)
            self.locref_map.append(locref_map)
            self.lowres_locref_map.append(lowres_locref)
            if self.im[i].ndim < 3:
                self.im[i] = self.im[i][np.newaxis, ...]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, item):
        """
        Input :-
            item: scalar integer.
        Output (dict):
            image: torch float32 tensor of size ncolors x height x width
            landmarks: nlandmarks x 2 float ndarray
            heatmaps: torch float32 tensor of size nlandmarks x height x width
            id: scalar integer, contains item
        """
        # Data augmentation
        im, lm_heatmap, _ = random_rotate_and_resize([self.im[item]],
                                                     [self.landmark_heatmaps[item]],
                                                     xy=self.img_xy,
                                                     scale_range=self.scale,
                                                     do_flip=self.flip,
                                                     rotation=self.rotation)
        lm = heatmap2landmarks(lm_heatmap)

        # Return tensors only
        im = torch.tensor(np.squeeze(im, axis=0), dtype=torch.float32)
        lm_heatmap = torch.tensor(np.squeeze(lm_heatmap, axis=0), dtype=torch.float32)
        lm = torch.tensor(np.squeeze(lm, axis=0), dtype=torch.float32)
        im = torchvision.transforms.Resize(112)(im)
        im = torchvision.transforms.Normalize(mean=0.2818, std=0.2170)(im)
        lm_heatmap = torchvision.transforms.Resize(112)(lm_heatmap)

        locref_map = torch.tensor(self.locref_map[item], dtype=torch.float32)
        lowres_locref_map = torch.tensor(self.lowres_locref_map[item], dtype=torch.float32)
        features = {'image': im,
                    'landmarks': lm,
                    'heatmap': lm_heatmap,
                    'lowres_locref_map': lowres_locref_map,
                    'locref_map': locref_map,
                    'id': item}
        return features

    @staticmethod
    def get_landmarks(d, i=None):
        if i is None:
            locs = d['landmarks']
        else:
            locs = d['landmarks'][i]
        return locs

    @staticmethod
    def get_heatmap_image(d, i, cmap='jet', colors=None):
        if i is None:
            hm = d['heatmap']
        else:
            hm = d['heatmap'][i, ...]
        hm = hm.numpy()
        im = heatmap2image(hm, cmap=cmap, colors=colors)
        return im

    @staticmethod
    def get_image(d, i=None):
        """
        static function, used for visualization
        COCODataset.get_image(d,i=None)
        Returns an image usable with plt.imshow()
        Inputs:
        d: if i is None, item from a COCODataset.
        if i is a scalar, batch of examples from a COCO Dataset returned
        by a DataLoader.
        i: Index of example into the batch d, or None if d is a single example
        Returns the ith image from the patch as an ndarray plottable with
        plt.imshow()
        """
        if i is None:
            im = np.squeeze(np.transpose(d['image'].numpy(), (1, 2, 0)), axis=2)
        else:
            im = np.squeeze(np.transpose(d['image'][i, ...].numpy(), (1, 2, 0)), axis=2)
        return im

    def normalize99(self, img):
        """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
        X = img.copy()
        x01 = np.percentile(X, 1)
        x99 = np.percentile(X, 99)
        X = (X - x01) / (x99 - x01)
        return X

    def make_heatmap_target(self, locs, imsz, scale=None):
        """
        Inputs:
            locs: nlandmarks x 2 ndarray
            imsz: image shape
        Returns:
            target: torch tensor of size nlandmarks x imsz[0] x imsz[1]
        """
        # allocate the tensor
        target = torch.zeros((locs.shape[0], imsz[0], imsz[1]), dtype=torch.float32)
        # loop through landmarks
        for i in range(locs.shape[0]):
            # location of this landmark to the nearest pixel
            if ~np.isnan(locs[i, 0]) and ~np.isnan(locs[i, 1]):
                # location of this landmark to the nearest pixel
                if scale is None:
                    x = int(np.round(locs[i, 0]))  # losing sub-pixel accuracy
                    y = int(np.round(locs[i, 1]))
                else:
                    x = int(np.round(locs[i, 0] / scale))  # losing sub-pixel accuracy
                    y = int(np.round(locs[i, 1] / scale))
                # edges of the Gaussian filter to place, minding border of image
                x0 = np.maximum(0, x - self.label_filter_r)
                x1 = np.minimum(imsz[1] - 1, x + self.label_filter_r)
                y0 = np.maximum(0, y - self.label_filter_r)
                y1 = np.minimum(imsz[0] - 1, y + self.label_filter_r)
                # crop filter if it goes outside of the image
                fil_x0 = self.label_filter_r - (x - x0)
                fil_x1 = self.label_filter_d - (self.label_filter_r - (x1 - x))
                fil_y0 = self.label_filter_r - (y - y0)
                fil_y1 = self.label_filter_d - (self.label_filter_r - (y1 - y))
                # copy the filter to the relevant part of the heatmap image
                if len(np.arange(y0, y1 + 1)) != len(np.arange(fil_y0, fil_y1 + 1)) or len(
                        np.arange(x0, x1 + 1)) != len(np.arange(fil_x0, fil_x1 + 1)):
                    target[i, y0:y1 + 1, x0:x1 + 1] = self.label_filter[fil_y0:fil_y0 + len(np.arange(y0, y1 + 1)),
                                                      fil_x0:fil_x0 + len(np.arange(x0, x1 + 1))]
                else:
                    target[i, y0:y1 + 1, x0:x1 + 1] = self.label_filter[fil_y0:fil_y1 + 1, fil_x0:fil_x1 + 1]
        return target

    def init_label_filter(self):
        """
        init_label_filter(self)
        Helper function
        Create a Gaussian filter for the heatmap target output
        """
        # radius of the filter
        self.label_filter_r = max(int(round(3 * self.label_sigma)), 1)
        # diameter of the filter
        self.label_filter_d = 2 * self.label_filter_r + 1

        # allocate
        self.label_filter = np.zeros([self.label_filter_d, self.label_filter_d])
        # set the middle pixel to 1. 
        self.label_filter[self.label_filter_r, self.label_filter_r] = 1.
        # blur with a Gaussian
        self.label_filter = cv2.GaussianBlur(self.label_filter, (self.label_filter_d, self.label_filter_d),
                                             self.label_sigma)
        # normalize
        self.label_filter = self.label_filter / np.max(self.label_filter)
        # convert to torch tensor
        self.label_filter = torch.from_numpy(self.label_filter)
