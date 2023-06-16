import os
import random
from datetime import datetime
from ptflops import get_model_complexity_info

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from thop import clever_format

from config import cfg
from loading_data import loading_data
from utils import *
from timer import Timer
import pdb

exp_name = cfg.TRAIN.EXP_NAME
log_txt = cfg.TRAIN.EXP_LOG_PATH + '/' + exp_name + '.txt'
writer = SummaryWriter(cfg.TRAIN.EXP_PATH + '/' + exp_name)

pil_to_tensor = standard_transforms.ToTensor()
train_loader, val_loader, restore_transform = loading_data()


def main():

    cfg_file = open('./config.py', "r")
    cfg_lines = cfg_file.readlines()

    with open(log_txt, 'a') as f:
        f.write(''.join(cfg_lines) + '\n\n\n\n')
    if len(cfg.TRAIN.GPU_ID) == 1:
        torch.cuda.set_device(cfg.TRAIN.GPU_ID[0])
    torch.backends.cudnn.benchmark = True

    net = []

    net = selectModel(cfg.MODEL.NAME)

    # if cfg.TRAIN.STAGE == 'all':
    #     net = ENet(only_encode=False)
    #     if cfg.TRAIN.PRETRAINED_ENCODER != '':
    #         encoder_weight = torch.load(cfg.TRAIN.PRETRAINED_ENCODER)
    #         del encoder_weight['classifier.bias']
    #         del encoder_weight['classifier.weight']
    #         # pdb.set_trace()
    #         net.encoder.load_state_dict(encoder_weight)

    if len(cfg.TRAIN.GPU_ID) > 1:
        net = torch.nn.DataParallel(net, device_ids=cfg.TRAIN.GPU_ID).cuda()
    else:
        net = net.cuda()

    net.train()

    if cfg.DATA.NUM_CLASSES == 1:
        criterion = torch.nn.BCEWithLogitsLoss().cuda()  # Binary Classification
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()  # Multiclass Classification

    optimizer = optim.Adam(net.parameters(), lr=cfg.TRAIN.LR,
                           weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = StepLR(
        optimizer, step_size=cfg.TRAIN.NUM_EPOCH_LR_DECAY, gamma=cfg.TRAIN.LR_DECAY)
    _t = {'train time': Timer(), 'val time': Timer()}
    print("base line validation:")
    validate(val_loader, net, criterion, optimizer, -1, restore_transform)

    print("===========================================================")
    print(
        f'start training at=> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    all_iou = []
    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        _t['train time'].tic()
        train(train_loader, net, criterion, optimizer, epoch)
        _t['train time'].toc(average=False)
        print('Epoch {} - Training time: {:.2f}s'.format(epoch +
              1, _t['train time'].diff))

        _t['val time'].tic()
        iu = validate(val_loader, net, criterion,
                      optimizer, epoch, restore_transform)
        all_iou.append(iu)
        _t['val time'].toc(average=False)
        print('Epoch {} - Validation time: {:.2f}s'.format(epoch +
              1, _t['val time'].diff))

    result = calculate_average(all_iou)
    print("Average IoU:", result)
    save_model_with_timestamp(net, cfg.TRAIN.MODEL_SAVE_PATH)
    macs, params = count_your_model(net)
    macs, params = clever_format([macs * 2 , params], "%.3f") #converted macs into flops and it only shows 3 decimal points.
    print('{:<30}  {:<8}'.format('GFLOPS: ',macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def train(train_loader, net, criterion, optimizer, epoch):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        if cfg.DATA.NUM_CLASSES == 1:
            loss = criterion(outputs, labels.unsqueeze(1).float())
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()


def validate(val_loader, net, criterion, optimizer, epoch, restore):
    cls_ius = np.zeros(4)   #Createing an array of zeroes to add all the ius for each class
    net.eval()
    criterion.cpu()
    input_batches = []
    output_batches = []
    label_batches = []
    iou_ = 0.0
    with torch.no_grad():
        for vi, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            if cfg.DATA.NUM_CLASSES == 1:
                # For binary classification
                outputs[outputs > 0.5] = 1
                outputs[outputs <= 0.5] = 0
                iou_ += calculate_mean_iu([outputs.squeeze_(1).data.cpu().numpy()],
                                          [labels.data.cpu().numpy()],
                                          2)
            else:
                # For multi-classification ???
                _, predicted = torch.max(outputs, 1)
                pred = predicted.data.cpu().numpy()
                leb = labels.data.cpu().numpy()
                res, cls_iu = scores([leb],[pred], cfg.DATA.NUM_CLASSES)
                iou_ += res['Mean IoU : \t']
                # print(cls_iu)
                for i in range(4):
                    cls_ius[i] += cls_iu[i]         #sum the iu of all classes seperately
    if cfg.DATA.NUM_CLASSES != 1:
        cls_ius = cls_ius / len(val_loader)  #Out of loop ---> calcualte average by deviding by the entire loop length
        print("------------------------------------------------------")
        print("|    paper   |   bottle   |  aluminium  |   Nylon    |")
        print("|   %.4f   |   %.4f   |   %.4f    |   %.4f   |" % (cls_ius[0],cls_ius[1],cls_ius[2],cls_ius[3]) )       #fancy printing the ius seperately for each class
        print("------------------------------------------------------")
    mean_iu = iou_/len(val_loader)
    print('[mean iu %.4f]' % (mean_iu))
    net.train()
    criterion.cuda()
    return mean_iu


if __name__ == '__main__':
    main()
