from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import cv2


def save_images(save_dir, visuals, image_name, image_size, prob_map):
    """save images to disk"""
    image_name = image_name[0]
    oriSize = (image_size[0].item(), image_size[1].item())
    palet_file = 'datasets/palette.txt'
    impalette = list(np.genfromtxt(palet_file, dtype=np.uint8).reshape(3*256))

    for label, im_data in visuals.items():
        if label == 'output':
            if prob_map:
                im = tensor2confidencemap(im_data)
                im = cv2.resize(im, oriSize)
                cv2.imwrite(os.path.join(save_dir, image_name[:-10]+'road_'+image_name[-10:]), im)
            else:
                im = tensor2labelim(im_data, impalette)
                im = cv2.resize(im, oriSize)
                cv2.imwrite(os.path.join(save_dir, image_name), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

def tensor2im(input_image, imtype=np.uint8):
    """Converts a image Tensor into an image array (numpy)"""
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))* 255.0
    return image_numpy.astype(imtype)

def tensor2labelim(label_tensor, impalette, imtype=np.uint8):
    """Converts a label Tensor into an image array (numpy),
    we use a palette to color the label images"""
    if len(label_tensor.shape) == 4:
        _, label_tensor = torch.max(label_tensor.data.cpu(), 1)

    label_numpy = label_tensor[0].cpu().float().detach().numpy()
    label_image = Image.fromarray(label_numpy.astype(np.uint8))
    label_image = label_image.convert("P")
    label_image.putpalette(impalette)
    label_image = label_image.convert("RGB")
    return np.array(label_image).astype(imtype)

def tensor2confidencemap(label_tensor, imtype=np.uint8):
    """Converts a prediction Tensor into an image array (numpy),
    we output predicted probability maps for kitti submission"""
    softmax_numpy = label_tensor[0].cpu().float().detach().numpy()
    softmax_numpy = np.exp(softmax_numpy)
    label_image = np.true_divide(softmax_numpy[1], softmax_numpy[0] + softmax_numpy[1])
    label_image = np.floor(255 * (label_image - label_image.min()) / (label_image.max() - label_image.min()))
    return np.array(label_image).astype(imtype)


def print_current_losses(epoch, i, losses, t, t_data):
    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)
    print(message)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def confusion_matrix(x, y, n, ignore_label=None, mask=None):
    if mask is None:
        mask = np.ones_like(x) == 1
    k = (x >= 0) & (y < n) & (x != ignore_label) & (mask.astype(np.bool))
    return np.bincount(n * x[k].astype(int) + y[k], minlength=n**2).reshape(n, n)

def getScores(conf_matrix):
    if conf_matrix.sum() == 0:
        return 0, 0, 0, 0, 0
    with np.errstate(divide='ignore',invalid='ignore'):
        globalacc = np.diag(conf_matrix).sum() / np.float(conf_matrix.sum())
        classpre = np.diag(conf_matrix) / conf_matrix.sum(0).astype(np.float)
        classrecall = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float)
        IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(np.float)
        pre = classpre[1]
        recall = classrecall[1]
        iou = IU[1]
        F_score = 2*(recall*pre)/(recall+pre)
    return globalacc, pre, recall, F_score, iou
