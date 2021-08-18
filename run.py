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

        train_ds = ImageDataset(train_, transforms=get_train_transforms(config["img_size"]))
        valid_ds = ImageDataset(valid_, transforms=get_valid_transforms(config["img_size"]))

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
        print(config.model_name)
        model = ImageModel(
            config,
            device
        )

        model.eval()
        model = model.to(device)

        # optimizer = torch.optim.Adam(model.parameters(), lr=config['schedular_params']['lr_start'], weight_decay=config['weight_decay'])
        base_optimizer = torch.optim.Adam
        optimizer = SAM(model.parameters(), base_optimizer, lr=config['schedular_params']['lr_start'], weight_decay=config['weight_decay'])
        scheduler = MyScheduler(optimizer, **config["schedular_params"])
        #er = EarlyStopping(config['patience'])

        loss_tr = nn.BCEWithLogitsLoss().to(device)
        loss_vl = nn.BCEWithLogitsLoss().to(device)

        for epoch in range(config["epochs"]):
            scheduler.step()
            loss_train = train_func(train_loader, model, device, loss_tr, optimizer, debug=config["debug"], sam=True)
            loss_valid, accuracy = valid_func(valid_loader, model, device, loss_tr)
            logging.debug(f"{epoch}epoch : loss_train > {loss_train} looss_valid > {loss_valid}")

            print("train_loss : ", loss_train)
            print("valid_loss : ", loss_valid)

            torch.save(model.state_dict(), f'save/{config["model_name"]}_epoch{epoch}_fold{fold}.pth')

        del model, train_loader, valid_loader, optimizer, scheduler

if __name__ == '__main__':
    main()
