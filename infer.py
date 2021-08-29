import os
import argparse
import datetime
import logging
import json
import pandas as pd
import os
import torch
from torch import nn
#import cudf

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

    # log file
    # now = datetime.datetime.now()
    # logging.basicConfig(
    #     filename='./logs/infer_' + config["model_name"] + '_'+ '{0:%Y%m%d%H%M%S}.log'.format(now), level=logging.DEBUG
    # )
    # logging.debug('infer')
    # logging.debug('date : {0:%Y,%m/%d,%H:%M:%S}'.format(now))
    # log_list = ["img_size", "train_bs", "monitor"]

    # for log_c in log_list:
    #     logging.debug(f"{log_c} : {config[log_c]}")
    from logging import getLogger, StreamHandler,FileHandler, Formatter, DEBUG, INFO
    logger = getLogger("logger")    #logger名loggerを取得
    logger.setLevel(DEBUG)  #loggerとしてはDEBUGで
    #handler1を作成
    handler_stream = StreamHandler()
    handler_stream.setLevel(DEBUG)
    handler_stream.setFormatter(Formatter("%(asctime)s: %(message)s"))
    #handler2を作成
    config_filename = os.path.basename(options.config).split(".")[0]
    handler_file = FileHandler(filename=f'./logs/infer_{config_filename}.log')
    handler_file.setLevel(DEBUG)
    handler_file.setFormatter(Formatter("%(asctime)s: %(message)s"))
    #loggerに2つのハンドラを設定
    logger.addHandler(handler_stream)
    logger.addHandler(handler_file)

    # train 用 df の作成
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    df, image_paths = read_dataset()
    df_test, test_paths = read_test_dataset()

    train_df["image_path"] = image_paths
    train_df["label"] = df["target"]
    # train_df["id"] = df["id"]

    test_df["image_path"] = test_paths
    test_df["label"] = df_test["target"]
    # test_df["id"] = df_test["id"]

    del df

    # le = LabelEncoder()
    # train_df.label = le.fit_transform(train_df.label)

    # modelの作成
    seed_everything(config['seed'])
    device = torch.device(f"cuda:{options.device}")
    # n_used_epoch = 2
    cols = ["oof", "oof_best_score", "oof_best_loss", "label"]
    oof_df = pd.DataFrame(index=[i for i in range(train_df.shape[0])], columns=cols)
    # oof_df["id"] = train_df.id
    oof_df["label"] = train_df.label
    oof_df["oof"] = 0
    print(oof_df.shape)

    for best_type in ["best_score", "best_loss"]:

        print(f'inference type {best_type} start')
        model = ImageModel(
                    1,
                    config["model_name"],
                    config["model_type"],
                    config["fc_dim"],
                    config["margin"],
                    config["scale"],
                    device,
                    training=False
                )

        model.eval()

        # dataset, dataloafer作成
        folds = StratifiedKFold(
                    n_splits=config['fold_num'],
                    shuffle=True,
                    random_state=config['seed']).split(np.arange(train_df.shape[0]),
                    train_df.label.values
                )

        test_preds = []
        val_preds = []
        valid_index = []

        for fold, (trn_idx, val_idx) in enumerate(folds):
            if fold > 0 and options.debug: # 時間がかかるので最初のモデルのみ
                break
            
            model.load_state_dict(torch.load(f'save/{config["model_name"]}_fold{fold}_{best_type}.pth'))
            model = model.to(device)

            valid_ = train_df.loc[val_idx,:].reset_index(drop=True)

            valid_ds = ImageDataset(valid_, transforms=get_valid_transforms(config["img_size"]))
            test_ds = ImageDataset(test_df, transforms=get_valid_transforms(config["img_size"]))

            valid_loader = torch.utils.data.DataLoader(
                valid_ds,
                batch_size=config["valid_bs"],
                num_workers=config["num_workers"],
                shuffle=False,
                pin_memory=True,
            )
            
            test_loader = torch.utils.data.DataLoader(
                test_ds,
                batch_size=config["valid_bs"],
                num_workers=config["num_workers"],
                shuffle=False,
                pin_memory=True,
            )

            valid_predictions = get_prediction(model, valid_loader, device).detach().cpu().numpy()
            print(valid_predictions.shape, valid_predictions.max(), valid_predictions.min(), valid_predictions.mean())
            test_prediction = get_prediction(model, test_loader, device).detach().cpu().numpy()
            val_preds.append(valid_predictions)
            test_preds.append(test_prediction)
            valid_index.append(val_idx)
        del model

        val_preds = np.concatenate(val_preds)
        valid_index = np.concatenate(valid_index)
        order = np.argsort(valid_index)
        oof_df["oof"] += val_preds[order]
        oof_df[f"oof_{best_type}"] = val_preds[order]
        score = roc_auc_score(oof_df.label, oof_df[f"oof_{best_type}"])
        logging.debug(f" type : {best_type}")
        logging.debug(f" CV_score : {score}")
        # logging.debug(f" scores : {scores.mean()}")

    del valid_loader, valid_predictions

    # submission
    sub = pd.read_csv("./data/input/sample_submission.csv")
    sub["target"] = np.mean(test_preds, axis=0)
    sub.to_csv(f"./data/output/{config_filename}.csv", index=False)
    sub["target"] = np.mean(test_preds[:fold], axis=0)
    sub.to_csv(f"./data/output/{config_filename}_best_score.csv", index=False)
    sub["target"] = np.mean(test_preds[fold:], axis=0)
    sub.to_csv(f"./data/output/{config_filename}_best_loss.csv", index=False)

    # oof
    oof_df["oof"] /= 2
    oof_df.loc[:, ["oof", "label"]].to_csv(f"./data/output/{config_filename}_oof.csv", index=False)
    oof_df.loc[:, ["oof_best_score", "label"]].to_csv(f"./data/output/{config_filename}_oof_best_score.csv", index=False)
    oof_df.loc[:, ["oof_best_loss", "label"]].to_csv(f"./data/output/{config_filename}_oof_best_loss.csv", index=False)

if __name__ == '__main__':
    main()