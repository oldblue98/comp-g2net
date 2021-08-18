import random
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def seed_everything(seed):
    "seed値を一括指定"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def read_dataset():
    df = pd.read_csv('./data/input/training_labels.csv')
    #df_cu = cudf.DataFrame(df)
    image_paths = "./data/input/train/" + df["id"].apply(lambda x:x[0]) + "/" + df["id"] + ".npy"
    #return df, df_cu, image_paths
    return df, image_paths

def read_test_dataset():
    df = pd.read_csv('./data/input/sample_submission.csv')
    #df_cu = cudf.DataFrame(df)
    image_paths = df['id'].apply(get_test_file_path)
    #return df, df_cu, image_paths
    return df, image_paths

def get_train_file_path(image_id):
    return "./data/input/train/{}/{}/{}/{}.npy".format(
        image_id[0], image_id[1], image_id[2], image_id)

def get_test_file_path(image_id):
    return "./data/input/test/{}/{}/{}/{}.npy".format(
        image_id[0], image_id[1], image_id[2], image_id)


def f1_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    return f1

def combine_predictions(row):
    x = np.concatenate([row['image_predictions'], row['text_predictions']])
    return ' '.join( np.unique(x) )

def get_img(path):
    """
    pathからimageの配列を得る
    """
    im_bgr = cv2.imread(path)
    if im_bgr is None:
        print(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb

def row_wise_f1_score(labels, preds):
    scores = []
    for label, pred in zip(labels, preds):
        n = len(np.intersect1d(label, pred))
        score = 2 * n / (len(label)+len(pred))
        scores.append(score)
    return scores, np.mean(scores)

