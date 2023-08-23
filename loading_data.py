import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import transforms as own_transforms
from resortit import resortit
from config import cfg
from torch.utils.data import ConcatDataset



def loading_data():
    mean_std = cfg.DATA.MEAN_STD
    if not cfg.TRAIN.PRETRAINING:
        if cfg.TRAIN.AUGMENTATION == "T1":
            train_simul_transform = own_transforms.Compose([
                own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0] / 0.875)),
                own_transforms.RandomCrop(cfg.TRAIN.IMG_SIZE),
                own_transforms.ColorJitter(),
                own_transforms.RandomHorizontallyFlip()
    ])
        if cfg.TRAIN.AUGMENTATION == "T2":
            train_simul_transform = own_transforms.Compose([
                own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0] / 0.875)),
                own_transforms.RandomCrop(cfg.TRAIN.IMG_SIZE),
                own_transforms.ColorJitter(),
                own_transforms.RandomHorizontallyFlip(),
                own_transforms.RandomResizedCrop(0.5)
    ])
        if cfg.TRAIN.AUGMENTATION == "T3":
            train_simul_transform = own_transforms.Compose([
                own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0] / 0.875)),
                own_transforms.RandomCrop(cfg.TRAIN.IMG_SIZE),
                own_transforms.ColorJitter(),
                own_transforms.RandomHorizontallyFlip(),
                own_transforms.RandomResizedCrop(0.75)
    ])
        else:
            train_simul_transform = own_transforms.Compose([
                own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0] / 0.875)),
                own_transforms.RandomCrop(cfg.TRAIN.IMG_SIZE),
                own_transforms.RandomHorizontallyFlip()
            ])
    if cfg.TRAIN.PRETRAINING:
        train_simul_transform0 = own_transforms.Compose([
            own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0]/ 0.875)),
            own_transforms.RandomCrop(cfg.TRAIN.IMG_SIZE),
        ])
        train_simul_transform1 = own_transforms.Compose([
            own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0]/ 0.875)),
            own_transforms.Rotate_90(),
            own_transforms.RandomCrop(cfg.TRAIN.IMG_SIZE)
        ])
        train_simul_transform2 = own_transforms.Compose([
            own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0]/ 0.875)),
            own_transforms.Rotate_180(),
            own_transforms.RandomCrop(cfg.TRAIN.IMG_SIZE)
        ])
        train_simul_transform3 = own_transforms.Compose([
            own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0]/ 0.875)),
            own_transforms.Rotate_270(),
            own_transforms.RandomCrop(cfg.TRAIN.IMG_SIZE)
    ])
    val_simul_transform = own_transforms.Compose([
        own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0] / 0.875)),
        own_transforms.CenterCrop(cfg.TRAIN.IMG_SIZE)
    ])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = standard_transforms.Compose([
        own_transforms.MaskToTensor(),
        own_transforms.ChangeLabel(cfg.DATA.IGNORE_LABEL, cfg.DATA.NUM_CLASSES - 1)
    ])
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    if cfg.TRAIN.PRETRAINING:
        train_set = []
    
        for i in range(4):
            match i:
                case 0:
                    set = resortit('train', simul_transform=train_simul_transform0, transform=img_transform,
                           target_transform=target_transform)
                    train_set.append((set))
                case 1:
                    set = resortit('train', simul_transform=train_simul_transform1, transform=img_transform,
                           target_transform=target_transform)
                    train_set.append(set)
                case 2:
                    set = resortit('train', simul_transform=train_simul_transform2, transform=img_transform,
                           target_transform=target_transform)
                    train_set.append(set)
                case 3:
                    set = resortit('train', simul_transform=train_simul_transform3, transform=img_transform,
                            target_transform=target_transform)
                    train_set.append(set)
            train_set = ConcatDataset(train_set)
    else:
        train_set = resortit('train', simul_transform=train_simul_transform, transform=img_transform,
                           target_transform=target_transform)
  
    train_loader = DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=4, shuffle=True)
    val_set = resortit('val', simul_transform=val_simul_transform, transform=img_transform,
                         target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=cfg.VAL.BATCH_SIZE, num_workers=4, shuffle=False)

    return train_loader, val_loader, restore_transform
