import torchvision.transforms as std_trans  # standard_transforms
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from config import cfg
from resortit import resortit
import transforms as o_t  # own_transformer


def loading_data():
    mean_std = cfg.DATA.MEAN_STD
    image_size = cfg.TRAIN.IMG_SIZE
    ignore_label = cfg.DATA.IGNORE_LABEL
    number_classes = cfg.DATA.NUM_CLASSES

    val_sim_t = o_t.Compose([o_t.Scale(int(image_size[0] / 0.875)), o_t.CenterCrop(image_size)])
    img_t = std_trans.Compose([std_trans.ToTensor(), std_trans.Normalize(*mean_std)])
    target_t = std_trans.Compose([o_t.MaskToTensor(), o_t.ChangeLabel(ignore_label, number_classes - 1)])
    restore_t = std_trans.Compose([o_t.DeNormalize(*mean_std), std_trans.ToPILImage()])

    if cfg.TRAIN.PRETRAINING:
        rotations = [o_t.Rotate_none, o_t.Rotate_90, o_t.Rotate_180, o_t.Rotate_270]
        train_set = []
        for rotation in rotations:
            train_sim_t = o_t.Compose([o_t.Scale(int(image_size[0] / 0.875)), rotation(), o_t.RandomCrop(image_size)])
            rotate_item = resortit('train', simul_transform=train_sim_t, transform=img_t, target_transform=target_t)
            train_set.append(rotate_item)
        train_set = ConcatDataset(train_set)
    else:
        augmentations = {"T1": [o_t.Scale(int(image_size[0] / 0.875)), o_t.RandomCrop(image_size), o_t.ColorJitter(),
                                o_t.RandomHorizontallyFlip()],
                         "T2": [o_t.Scale(int(image_size[0] / 0.875)), o_t.RandomRotate(), o_t.RandomCrop(image_size),
                                o_t.ColorJitter(),
                                o_t.RandomHorizontallyFlip()],
                         "T3": [o_t.Scale(int(image_size[0] / 0.875)), o_t.RandomRotate(), o_t.RandomResizedCrop(),
                                o_t.ColorJitter(),
                                o_t.RandomHorizontallyFlip()],
                         None: [o_t.Scale(int(image_size[0] / 0.875)), o_t.RandomCrop(image_size),
                                o_t.RandomHorizontallyFlip()]}

        train_sim_t = o_t.Compose(augmentations[cfg.TRAIN.AUGMENTATION])
        train_set = resortit('train', simul_transform=train_sim_t, transform=img_t, target_transform=target_t)

    val_set = resortit('val', simul_transform=val_sim_t, transform=img_t, target_transform=target_t)
    train_loader = DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.VAL.BATCH_SIZE, num_workers=4, shuffle=False)

    return train_loader, val_loader, restore_t
