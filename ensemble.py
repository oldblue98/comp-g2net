import argparse
import json
import os
import datetime

import lightgbm as lgb

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss, roc_auc_score
from torch.nn.functional import linear

# 引数で config の設定を行う
parser = argparse.ArgumentParser()
# parser.add_argument('--config', default='./configs/default.json')
parser.add_argument('--fold_num', default=2)
parser.add_argument('--seed', default=42)

# parser.add_argument('--metric', default='mean')
parser.add_argument('ensemble_name', type=str)

options = parser.parse_args()
# CFG = json.load(open(options.config))

ensemble_name = options.ensemble_name
# metric = options.metric

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_class': 1,
    'learning_rate': 0.01,
    'max_depth': 4,
    'num_leaves':3,
    'lambda_l2' : 0.3,
    'num_iteration': 1000,
    "min_data_in_leaf":1,
    'verbose': 0
}

logistic_params = {

}

linear_params = {

}

oof_path = [
    "tf_efficientnet_b7_ns_ver3_oof.csv",
    "tf_efficientnet_b7_ns_ver4_oof.csv",
    "tf_efficientnet_b7_ns_ver7_oof.csv",
]

test_path = [
    "tf_efficientnet_b7_ns_ver3.csv",
    "tf_efficientnet_b7_ns_ver4.csv",
    "tf_efficientnet_b7_ns_ver7.csv",
]

data_path = "./data/output/"

# logger の設定
from logging import getLogger, StreamHandler,FileHandler, Formatter, DEBUG, INFO
logger = getLogger("logger")    #logger名loggerを取得
logger.setLevel(DEBUG)  #loggerとしてはDEBUGで
#handler1を作成
handler_stream = StreamHandler()
handler_stream.setLevel(DEBUG)
handler_stream.setFormatter(Formatter("%(asctime)s: %(message)s"))
#handler2を作成
config_filename = os.path.splitext(os.path.basename(options.config))[0]
handler_file = FileHandler(filename=f'./logs/ensemble_{ensemble_name}.log')
handler_file.setLevel(DEBUG)
handler_file.setFormatter(Formatter("%(asctime)s: %(message)s"))
#loggerに2つのハンドラを設定
logger.addHandler(handler_stream)
logger.addHandler(handler_file)


class LightGBM():
    def __init__(self, params):
        self.params = params

    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test):
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
        model = lgb.train(
            self.params,lgb_train, 
            valid_sets=lgb_valid,
            num_boost_round=1000,
            early_stopping_rounds=100
            )
        preds_val = model.predict(X_valid, num_iteration=model.best_iteration)
        preds_test = model.predict(X_test, num_iteration=model.best_iteration)
        return preds_val, preds_test

class LogisticWrapper():
    def __init__(self, params):
        self.params = params

    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        preds_val = model.predict_proba(X_valid)
        preds_test = model.predict_proba(X_test)
        return preds_val, preds_test

class Linear():
    def __init__(self, params):
        self.params = params

    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test):
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds_val = model.predict(X_valid)
        preds_test = model.predict(X_test)
        return preds_val, preds_test


class meanWrapper():
    def __init__(self):
        pass

    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test):
        preds_val = np.mean(X_valid, axis=1)
        preds_test = np.mean(X_test, axis=1)
        return preds_val, preds_test

class Identfy():
    def __init__(self):
        pass

    def train_and_predict(self, X_train, X_valid, y_train, y_valid, X_test):
        return np.array(y_valid), np.array(X_test)


def load_oof_df(path):
    oof_df = pd.concat([pd.read_csv(data_path + p).drop(["label"], axis=1).rename(columns=lambda s: s + p) for p in path], axis=1)
    label = pd.read_csv(data_path + path[0]).loc[:, ["label"]]
    return oof_df, label

def load_test_df(path):
    test_df = pd.concat([pd.read_csv(data_path + p).loc[:, ["target"]].rename(columns=lambda s: s + p) for p in path], axis=1)
    return test_df


cols = ["0", "1"]
def main():
    oof_df, oof_label = load_oof_df(oof_path)
    test_df = load_test_df(test_path)

    folds = StratifiedKFold(n_splits=options["fold_num"], shuffle=True, random_state=options["seed"])
    # folds = GroupKFold(n_splits=CFG['fold_num']).split(np.arange(train.shape[0]), groups=train.id.values)

    for metric in ["lgb", "logistic", "mean", "linear"]:
        test_preds = []
        val_preds = []

        if metric == "lgb":
            model = LightGBM(params)
        elif metric == "logistic":
            model = LogisticWrapper(logistic_params)
        elif metric == "mean":
            model = meanWrapper()
        elif metric == "linear":
            model = Linear(linear_params)

        for fold, (tr_idx, val_idx) in enumerate(folds.split(np.arange(oof_df.shape[0]), oof_label.values)):
            X_train, X_valid = oof_df.iloc[tr_idx, :], oof_df.iloc[val_idx, :]
            y_train, y_valid = oof_label.iloc[tr_idx], oof_label.iloc[val_idx]
            
            y_pred_valid, y_pred_test = model.train_and_predict(X_train, X_valid, y_train, y_valid, test_df)
            # 結果を保存
            test_preds.append(y_pred_test)
            val_preds.append(y_pred_valid)

            # print(f'y_valid,shape : {y_valid.shape}, y_pred_valid : {y_pred_valid.shape}')
            # print(f'X_valid[:5] : {X_valid[-50:-1]}, y_pred_valid[:5] : {y_pred_valid[-50:-1]}')
            # print(f'y_valid,label : {y_valid.value_counts("label")}, y_pred_valid.label : {np.argmax(y_pred_valid, axis=1).sum()}')
            # print(f'y_pred_valid.argmax : {np.argmax(y_pred_valid, axis=1)}')
        val_preds = np.concatenate(val_preds)
        valid_index = np.concatenate(val_idx)
        order = np.argsort(valid_index)
        val_preds = val_preds[order]
        test_preds = np.mean(test_preds, axis=0)

        # スコア
        loss = log_loss(oof_label.label, val_preds)
        acc = roc_auc_score(oof_label.label, val_preds)

        # loss, aucの記録
        logger.debug('===CV scores===')
        logger.debug(f'ensemble type : {metric}')
        logger.debug(f"log loss: {loss}")
        logger.debug(f"AUC : {acc}")

        # loss = sum(scores_loss) / len(scores_loss)
        # logger.debug('===CV scores loss===')
        # logger.debug(f'scores_loss:{scores_loss}\n mean_loss:{loss}')
        
        # acc = sum(scores_acc) / len(scores_acc)
        # logger.debug('===CV scores acc===')
        # logger.debug(f'scores_acc:{scores_acc}\n mean_acc:{acc}')

        # 予測結果を保存
        sub = pd.read_csv("./data/input/sample_submission.csv")
        sub['label'] = test_preds
        logger.debug(sub.value_counts("label"))
        sub.to_csv(f'data/output/sub_ensemble_{ensemble_name}_{metric}.csv', index=False)
        oof_df.iloc[:, 0] = val_preds
        oof_df.to_csv(f'data/output/ensemble_{ensemble_name}_{metric}_oof.csv', index=False)

if __name__ == '__main__':
    main()