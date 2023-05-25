from torch import nn, save
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import shutil
from config import cfg
from datetime import datetime


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
    sum_iu = 0
    for i in range(num_classes):
        n_ii = t_i = sum_n_ji = 1e-9
        for p, gt in zip(predictions, gts):
            n_ii += np.sum(gt[p == i] == i)
            t_i += np.sum(gt == i)
            sum_n_ji += np.sum(p == i)
        sum_iu += float(n_ii) / (t_i + sum_n_ji - n_ii)
    mean_iu = sum_iu / num_classes
    return mean_iu


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
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
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
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall Acc: \t': acc,
            'Mean Acc : \t': acc_cls,
            'FreqW Acc : \t': fwavacc,
            'Mean IoU : \t': mean_iu, }, cls_iu


def calculate_average(numbers):
    if not numbers:
        return None  # Return None for an empty array

    total = sum(numbers)
    average = total / len(numbers)
    return average


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
