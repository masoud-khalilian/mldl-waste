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
import pdb
import numpy as np

from config import cfg
from loading_data import loading_data
from utils import *
from timer import Timer

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

    teacher_net = []  # for knowledge distillation
    teacher_net = selectModel('bisenet-resnet18')
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

    if cfg.TRAIN.USE_DISTILLATION:
        bisenet_wight = torch.load(cfg.TRAIN.TEACHER_PATH)
        teacher_net.load_state_dict(bisenet_wight)
        teacher_net = net.cuda()
        teacher_net.eval()

    criterion = get_criterion(num_classes=cfg.DATA.NUM_CLASSES, loss_func=cfg.TRAIN.MULTI_CLASS_LOSS)
    print('criterion', criterion)
    optimizer = optim.Adam(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.NUM_EPOCH_LR_DECAY, gamma=cfg.TRAIN.LR_DECAY)

    _t = {'train time': Timer(), 'val time': Timer()}
    print("base line validation:")
    validate(val_loader, net, criterion, optimizer, -1, restore_transform)
    print("===========================================================")
    print(f'start training at=> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        _t['train time'].tic()

        if cfg.TRAIN.USE_DISTILLATION is not True:
            train(train_loader, net, criterion, optimizer, epoch)
        else:
            train_knowledge_distillation(train_loader, net, teacher_net, criterion, optimizer, epoch)

        scheduler.step()
        _t['train time'].toc(average=False)
        print('Epoch {} - Training time: {:.2f}s'.format(epoch + 1, _t['train time'].diff))
        if(epoch % cfg.VALIDATION_CYCLE == 0 or epoch > 180):
            _t['val time'].tic()
            validate(val_loader, net, criterion, optimizer, epoch, restore_transform)
            _t['val time'].toc(average=False)
            print('Epoch {} - Validation time: {:.2f}s'.format(epoch + 1, _t['val time'].diff))

    save_model_with_timestamp(net, cfg.TRAIN.MODEL_SAVE_PATH)
    macs, params = count_your_model(net)
    # converted macs into flops and it only shows 3 decimal points.
    macs, params = clever_format([macs * 2, params], "%.3f")
    print('{:<30}  {:<8}'.format('GFLOPS: ', macs))
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
    net.eval()
    criterion.cpu()
    iou_ = 0.0
    cls_ius = [0.0] * cfg.DATA.NUM_CLASSES
    prediction = np.zeros(0)
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
                prediction = outputs.squeeze_(1).data.cpu().numpy()
                res, _ = calculate_mean_iu(prediction, labels.data.cpu().numpy(), 2)
                iou_ += res
            else:
                # For multi-classification
                prediction = outputs.argmax(dim=1).data.cpu().numpy()
                res, cls_iu = calculate_mean_iu(prediction, labels.data.cpu().numpy(), cfg.DATA.NUM_CLASSES)
                iou_ += res
                cls_ius = [sum(x) for x in zip(cls_ius, cls_iu)]

            if epoch == 1 or epoch == 99:
                if vi == 0:
                    save_images(inputs.data.cpu().numpy(), prediction, labels.data.cpu().numpy(), epoch, './images')

    if cfg.DATA.NUM_CLASSES == 1:
        print('[mean iu %.4f]' % (iou_ / len(val_loader)))
    else:
        mean_ius = [x / len(val_loader) for x in cls_ius]
        print("-------------------------------------------------------------------")
        print("|     none   |    paper   |   bottle    | aluminium  |   Nylon    |")
        print("|   %.4f   |   %.4f   |   %.4f    |   %.4f   |   %.4f   |" % (
            mean_ius[0], mean_ius[1], mean_ius[2], mean_ius[3], mean_ius[4]))
        print("-------------------------------------------------------------------")
        print('[mean iu %.4f]' % (iou_ / len(val_loader)))
    net.train()
    criterion.cuda()


def train_knowledge_distillation(train_loader, studnet_net, teacher_net, criterion, optimizer, epoch):
    T = 2
    soft_target_loss_weight = 0.50
    ce_loss_weight = 0.50
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_logits = teacher_net(inputs)
        student_logits = studnet_net(inputs)

        # Soften the student logits by applying softmax first and log() second
        soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)
        soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T ** 2)

        label_loss = criterion(student_logits, labels)
        # if cfg.DATA.NUM_CLASSES == 1:
        #     loss = criterion(outputs, labels.unsqueeze(1).float())
        # else:
        #     loss = criterion(outputs, labels)
        loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
