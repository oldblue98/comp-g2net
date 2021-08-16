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
    df = pd.read_csv('./data/input/train_labels.csv')
    #df_cu = cudf.DataFrame(df)
    image_paths = "./data/input/train/" + df["id"].apply(lambda x:x[0]) + "/" + df["id"] + ".npy"
    #return df, df_cu, image_paths
    return df, image_paths

def read_test_dataset():
    df = pd.read_csv('./data/input/sample_submission.csv')
    #df_cu = cudf.DataFrame(df)
    image_paths = "./data/input/test/" + df["id"].apply(lambda x:x[0]) + "/" + df["id"] + ".npy"
    #return df, df_cu, image_paths
    return df, image_paths

def getMetric(col):
    def rocscore(row):
        return roc_auc_score(row.target, row[col])
    return rocscore

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

def find_threshold(df, lower_count_thresh, upper_count_thresh, search_space, FEAS):
    '''
    Compute the optimal threshold for the given count threshold.
    '''
    score_by_threshold = []
    best_score = 0
    best_threshold = -1
    for i in tqdm(search_space):
        sim_thresh = i/100
        selection = ((FEAS@FEAS.T) > sim_thresh).cpu().numpy()
        matches = []
        oof = []
        for row in selection:
            oof.append(df.iloc[row].posting_id.tolist())
            matches.append(' '.join(df.iloc[row].posting_id.tolist()))
        tmp = df.groupby('label_group').posting_id.agg('unique').to_dict()
        df['target'] = df.label_group.map(tmp)
        scores, score = row_wise_f1_score(df.target, oof)
        df['score'] = scores
        df['oof'] = oof

        selected_score = df.query(f'count > {lower_count_thresh} and count < {upper_count_thresh}').score.mean()
        score_by_threshold.append(selected_score)
        if selected_score > best_score:
            best_score = selected_score
            best_threshold = i

    plt.title(f'Threshold Finder for count in [{lower_count_thresh},{upper_count_thresh}].')
    plt.plot(score_by_threshold)
    plt.axis('off')
    plt.show()
    print(f'Best score is {best_score} and best threshold is {best_threshold/100}')
