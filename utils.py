import torch
from torch import nn, save
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import shutil
from config import cfg
from datetime import datetime
from model_bisenet import BiSeNet
from model_enet import ENet
from model_icnet import ICNet
from torchvision.utils import save_image
from PIL import Image
from thop import profile
from torch.autograd import Variable


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
        'icnet': ICNet,  # Replace ICNet with the actual class name
        'bisenet': BiSeNet,  # Replace BiSeNet with the actual class name
        'enet': ENet  # Replace ENet with the actual class name
    }

    model = options[model_name]()
    return model


def save_binary_visualization(inputs, labels, outputs, index):
    os.makedirs("./vis", exist_ok=True)
    save_image(inputs, f'./vis/input_{index}.png')

    labels = add_arbitrary_rgb(labels)
    outputs = add_arbitrary_rgb(outputs)
    save_image(labels, f'./vis/labels_{index}.png')
    save_image(outputs, f'./vis/outputs_{index}.png')


def add_arbitrary_rgb(tensor):
    expanded_tensor = tensor.unsqueeze(1).repeat(1, 3, 1, 1)
    rgb_tensor = torch.zeros(
        16, 3, 224, 448, dtype=torch.float32, device='cuda')
    # Assign the original tensor as the red channel
    rgb_tensor[:, 0, :, :] = expanded_tensor[:, 0, :, :]
    # Assign the original tensor as the green channel
    rgb_tensor[:, 1, :, :] = expanded_tensor[:, 0, :, :]
    # Assign the original tensor as the blue channel
    rgb_tensor[:, 2, :, :] = expanded_tensor[:, 0, :, :]
    rgb_tensor[rgb_tensor != 0] = torch.clamp(
        rgb_tensor[rgb_tensor != 0] * 255, max=255)

    return rgb_tensor


def count_your_model(model):
    # your rule here

    input = torch.randn(16, 3, 224, 448, device='cuda:0')
    macs, params = profile(model, inputs=(input,))
    return macs, params
