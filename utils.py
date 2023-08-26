from matplotlib import pyplot as plt
from torch import nn, save
import torch
import torch.nn.functional as F
import numpy as np
import os
import shutil
from config import cfg
from datetime import datetime
from PIL import Image
from thop import profile

from models.bisenet.bisenet import BiSeNet
from models.bisenet.bisenetResnet import BiSeNetResnet
from models.bisenet.bisenet_f_f import BiSeNet_f_f
from models.bisenet.bisenet_f_h import BiSeNet_f_h
from models.bisenet.bisenet_h_f import BiSeNet_h_f
from models.bisenet.bisenet_h_h import BiSeNet_h_h
from models.enet.enet import ENet
from models.enet.enet_f_f import ENet_f_f
from models.enet.enet_f_h import ENet_f_h
from models.enet.enet_h_f import ENet_h_f
from models.enet.enet_h_h import ENet_h_h
from models.icnet.icnet import ICNet
from models.icnet.icnet_f_f import ICNet_f_f
from models.icnet.icnet_f_h import ICNet_f_h
from models.icnet.icnet_h_f import ICNet_h_f
from models.icnet.icnet_h_h import ICNet_h_h


def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        # kaiming is first name of author whose last name is 'He' lol
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()


def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def calculate_mean_iu(predictions, gts, num_classes):
    iou_classes = np.zeros(num_classes)

    for i in range(num_classes):
        n_ii = t_i = sum_n_ji = 1e-9

        for p, gt in zip(predictions, gts):
            if i in np.unique(gt):
                n_ii += np.sum((gt == i) & (p == i))
                t_i += np.sum(gt == i)
                sum_n_ji += np.sum(p == i)

        iou_classes[i] = n_ii / (t_i + sum_n_ji - n_ii)

    mean_iu = np.mean(iou_classes)
    return mean_iu, iou_classes


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


def rmrf_mkdir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)


def rm_file(path_file):
    if os.path.exists(path_file):
        os.remove(path_file)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(cfg.VIS.PALETTE_LABEL_COLORS)

    return new_mask


# ============================


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) +
                          hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = {i: iu[i] for i in range(n_class)}
    return {
        'Overall Acc: \t': acc,
        'Mean Acc : \t': acc_cls,
        'FreqW Acc : \t': fwavacc,
        'Mean IoU : \t': mean_iu
    }, cls_iu


def save_model_with_timestamp(model, model_dir):
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Generate the model file name with the timestamp
    model_name = f'trained_model_{timestamp}.pth'
    model_path = os.path.join(model_dir, model_name)

    # Save the model
    save(model.state_dict(), model_path)
    print("Trained model saved at:", model_path)

    # Calculate the model size in MB
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print("Model size:", model_size_mb, "MB")


def selectModel(model_name):
    options = {
        'icnet': ICNet,  # ICNet
        'icnet-f-f': ICNet_f_f,  # ICNet with fewer encoder and fewer decoder
        'icnet-f-h': ICNet_f_h,  # ICNet with fewer encoder and higher decoder
        'icnet-h-f': ICNet_h_f,  # ICNet with higher encoder and fewer decoder
        'icnet-h-h': ICNet_h_h,  # ICNet with higher encoder and higher decoder
        'bisenet': BiSeNet,  # BiSeNet with the actual class name
        'bisenet-f-f': BiSeNet_f_f,  # BiSeNet with fewer encoder and fewer decoder
        'bisenet-f-h': BiSeNet_f_h,  # BiSeNet with fewer encoder and higher decoder
        'bisenet-h-f': BiSeNet_h_f,  # BiSeNet with higher encoder and fewer decoder
        'bisenet-h-h': BiSeNet_h_h,  # BiSeNet with higher encoder and higher decoder
        'bisenet-resnet18': BiSeNetResnet,
        'enet': ENet,  # ENet with the actual class name
        'enet-f-f': ENet_f_f,  # ENet with fewer encoder and fewer decoder
        'enet-f-h': ENet_f_h,  # ENet with fewer encoder and higher decoder
        'enet-h-f': ENet_h_f,  # ENet with higher encoder and fewer decoder
        'enet-h-h': ENet_h_h,  # ENet with higher encoder and higher decoder
    }

    model = options[model_name]()
    return model


def count_your_model(model):
    # your rule here
    input = torch.randn(16, 3, 224, 448, device='cuda:0')
    macs, params = profile(model, inputs=(input,))
    return macs, params


def save_images(arr: np.ndarray, prediction, lables, epoch, path: str = './images'):
    if not os.path.exists('./images'):
        os.mkdir('./images')

    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    for i in range(arr.shape[0]):
        filename = f"{path}/epoch_{epoch}_input_{i}.png"
        plt.imsave(filename, arr[i].transpose(1, 2, 0))

    for i in range(lables.shape[0]):
        colorized_mask = colorize_mask(lables[i])
        colorized_mask.save(f"{path}/epoch_{epoch}_mask_{i}.png")

    for i in range(prediction.shape[0]):
        colorized_mask = colorize_mask(prediction[i])
        colorized_mask.save(f"{path}/epoch_{epoch}_prediction_{i}.png")
