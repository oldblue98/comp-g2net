import os
import argparse
import datetime
import logging
import json
from typing import Tuple
import pandas as pd
import os
import torch
from torch import nn
#import cudf

import sys

from torch.nn.modules import loss
sys.path.append('sam/')
from sam import SAM

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold

from src.augmentation import *
from src.dataset import *
from src.utils import *
from src.model import *

import warnings
warnings.filterwarnings('ignore')

def main():
    # config file upload
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/default.json')
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--device', default="0")
    options = parser.parse_args()
    config = json.load(open(options.config))


    from logging import getLogger, StreamHandler,FileHandler, Formatter, DEBUG, INFO
    logger = getLogger("logger")    #logger名loggerを取得
    logger.setLevel(DEBUG)  #loggerとしてはDEBUGで
    #handler1を作成
    handler_stream = StreamHandler()
    handler_stream.setLevel(DEBUG)
    handler_stream.setFormatter(Formatter("%(asctime)s: %(message)s"))
    #handler2を作成
    config_filename = os.path.basename(options.config).split(".")[0]
    handler_file = FileHandler(filename=f'./logs/train_{config_filename}.log')
    handler_file.setLevel(DEBUG)
    handler_file.setFormatter(Formatter("%(asctime)s: %(message)s"))
    #loggerに2つのハンドラを設定
    logger.addHandler(handler_stream)
    logger.addHandler(handler_file)

    # train 用 df の作成
    train_df = pd.DataFrame()
    df, image_paths = read_dataset()

    train_df["label"] = df["target"]
    train_df["image_path"] = image_paths

    if config["pseudo"] != "None":
        df, image_paths = read_pseudo_dataset(config["pseudo"])
        pseudo_df = pd.DataFrame()
        pseudo_df["label"] = df["target"]
        pseudo_df["image_path"] = image_paths
        pseudo_df = pseudo_df[(pseudo_df.label > 0.9) | (pseudo_df.label < 0.1)]
        pseudo_df["label"] = pseudo_df.apply(lambda x: x.label > 0.5, axis=1)
        print(f"pseudo_df.shape : {pseudo_df.shape}, {pseudo_df.columns}")

        train_df = pd.concat([train_df, pseudo_df], axis=0).reset_index(drop=True)
    print(f"train_df.shape : {train_df.shape}, {train_df.columns}")
    # le = LabelEncoder()
    # train_df.label = le.fit_transform(train_df.label)

    # modelの作成
    seed_everything(config['seed'])
    device = torch.device(f"cuda:{options.device}")

    # dataset, dataloafer作成
    folds = StratifiedKFold(
                n_splits=config['fold_num'],
                shuffle=True,
                random_state=config['seed']
                ).split(np.arange(train_df.shape[0]), train_df.label.values)

    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold > 0 and options.debug: # 時間がかかるので最初のモデルのみ
            break
        print(f'Training with fold {fold} started (train:{len(trn_idx)}, val:{len(val_idx)})')

        train_ = train_df.loc[trn_idx,:].reset_index(drop=True)
        valid_ = train_df.loc[val_idx,:].reset_index(drop=True)

        train_ds = ImageDataset(train_, config["qtransform_params"], transforms=get_train_transforms(config), image_type=config["image_type"])
        valid_ds = ImageDataset(valid_, config["qtransform_params"], transforms=get_valid_transforms(config), image_type=config["image_type"])

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=config["train_bs"],
            pin_memory=True, # faster and use memory
            drop_last=False,
            num_workers=config["num_workers"],
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=config["valid_bs"],
            num_workers=config["num_workers"],
            shuffle=False,
            pin_memory=True,
        )

        model = ImageModel(
            config,
            device
        )

        model.eval()
        model = model.to(device)

        def get_optimizer():
            if config["optimizer"] =="Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            elif config["optimizer"] == "SAM":
                base_optimizer = torch.optim.Adam
                optimizer = SAM(model.parameters(), base_optimizer, lr=config['lr'], weight_decay=config['weight_decay'])
            else:
                raise Exception('optimizer is not defined!')
            return optimizer

        def get_scheduler(optimizer):
            if config["scheduler"]=='CosineAnnealingLR':
                scheduler = CosineAnnealingLR(optimizer, last_epoch=-1, **config["CosineAnnealingLR"])
            elif config["scheduler"]=='CosineAnnealingWarmRestarts':
                scheduler = scheduler = CosineAnnealingWarmUpRestarts(optimizer, last_epoch=-1, **config["CosineAnnealingWarmRestarts"])
            else:
                raise Exception('scheduler is not defined!')
            return scheduler

        optimizer = get_optimizer()
        scheduler = get_scheduler(optimizer)

        #er = EarlyStopping(config['patience'])

        loss_tr = nn.BCEWithLogitsLoss().to(device)
        loss_vl = nn.BCEWithLogitsLoss().to(device)

        best_score = 0.
        best_loss = np.inf
        for epoch in range(config["epochs"]):
            logger.debug(f"lr : {scheduler.get_lr()[0]}")
            loss_train = train_func(train_loader, model, device, loss_tr, optimizer, debug=config["debug"], sam=config["optimizer"] == "SAM", mixup=config["mixup"])
            loss_valid, accuracy = valid_func(valid_loader, model, device, loss_tr, debug=config["debug"])
            logger.debug(f"{epoch+1}epoch : loss_train > {loss_train} loss_valid > {loss_valid} auc > {accuracy}")
            scheduler.step()
            
            if accuracy > best_score:
                best_score = accuracy
                logger.debug(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model")
                torch.save(model.state_dict(), f'save/{config_filename}_fold{fold}_best_score.pth')

            if loss_valid < best_loss:
                best_loss = loss_valid
                logger.debug(f"Epoch {epoch+1} - Save Best loss: {best_loss:.4f} Model")
                torch.save(model.state_dict(), f'save/{config_filename}_fold{fold}_best_loss.pth')

        del model, train_loader, valid_loader, optimizer, scheduler

if __name__ == '__main__':
    main()
