
import sys
sys.path.append('pytorch-image-models/')

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
import gc

from cuml.feature_extraction.text import TfidfVectorizer
from cuml.neighbors import NearestNeighbors
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import roc_auc_score

import timm
import math
import numpy as np
from tqdm import tqdm
import time
import warnings

from logging import getLogger, DEBUG, INFO, StreamHandler
logger = getLogger("logger")
handler_stream = StreamHandler()
handler_stream.setLevel(DEBUG)
logger.addHandler(handler_stream)

from src.utils import *

class ImageModel(nn.Module):
    def __init__(self, config, device,
                 use_fc=True, pretrained=True, training=True):

        super(ImageModel,self).__init__()
        print('Building Model Backbone for {} model'.format(config["model_name"]))
        self.slope = .1
        self.n = 16
        self.output_size = config["img_size"]
        self.r = 1

        self.model_name = config["model_name"]
        self.in_channels = config["in_channels"]
        self.n_classes = config["n_classes"]
        self.learn_resize = config["learn_resize"]


        self.backbone = timm.create_model(
            self.model_name, 
            pretrained=pretrained, 
            in_chans=self.in_channels, 
            drop_rate=config["drop_rate"], 
            drop_path_rate=config["drop_path_rate"]
        )

        if hasattr(self.backbone, "fc"):
            nb_ft = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif hasattr(self.backbone, "_fc"):
            nb_ft = self.backbone._fc.in_features
            self.backbone._fc = nn.Identity()

        elif hasattr(self.backbone, "classifier"):
            nb_ft = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        elif hasattr(self.backbone, "last_linear"):
            nb_ft = self.backbone.last_linear.in_features
            self.backbone.last_linear = nn.Identity()

        elif hasattr(self.backbone, "head"):
            if hasattr(self.backbone.head, "fc"):
                nb_ft = self.backbone.head.fc.in_features
                self.backbone.head.fc = nn.Identity()
            else:
                nb_ft = self.backbone.head.in_features
                self.backbone.head = nn.Identity()

            # self.backbone.head.global_pool = nn.Identity()

        print("nb_ft : ", nb_ft)
        if config["learn_resize"]:
            self.block1 = nn.Sequential(
                    nn.Conv2d(1, self.n, kernel_size=(7, 7), stride=(1,1), padding=(1, 1), bias=False),
                    nn.LeakyReLU(negative_slope=self.slope),
                    nn.Conv2d(self.n, self.n, kernel_size=(1, 1), stride=(1,1), padding=(1, 1), bias=False),
                    nn.LeakyReLU(negative_slope=self.slope),
                    nn.BatchNorm2d(self.n))
            self.block2 = nn.Sequential(
                    nn.Conv2d(self.n, self.n, kernel_size=(3, 3), stride=(1,1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(self.n),
                    nn.LeakyReLU(negative_slope=self.slope),
                    nn.Conv2d(self.n, self.n, kernel_size=(3, 3), stride=(1,1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(self.n))
            self.block3 = nn.Sequential(
                    nn.Conv2d(self.n, self.n, kernel_size=(3, 3), stride=(1,1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(self.n))
            self.block4 = nn.Sequential(
                    nn.Conv2d(self.n, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False))
                    
        self.fc = nn.Linear(nb_ft, self.n_classes)
        in_features = self.backbone.num_features
        print(f"{self.model_name}: {in_features}")

        # if model_type == 'res':
        #     final_in_features = self.backbone.fc.in_features
        #     self.backbone.fc = nn.Identity()
        #     self.backbone.global_pool = nn.Identity()

        # elif model_type == 'eff':
        #     final_in_features = self.backbone.classifier.in_features
        #     self.backbone.classifier = nn.Identity()
        #     self.backbone.global_pool = nn.Identity()

        # elif model_type == 'vit':
        #     final_in_features = self.backbone.head.in_features
        #     self.backbone.head = nn.Identity()
        #     self.backbone.global_pool = nn.Identity()

        # elif model_type == "fnet":
        #     final_in_features = self.backbone.head.fc.in_features
        #     self.backbone.head.fc = nn.Identity()
        #     self.backbone.head.global_pool = nn.Identity()

        # self.pooling =  nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc
        self.training = training

        # self.dropout = nn.Dropout(p=0.0)
        # self.fc = nn.Linear(in_features, fc_dim)
        # self.fc_ = nn.Linear(fc_dim, n_classes)
        # self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        # self.final = 

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        # nn.init.constant_(self.bn.weight, 1)
        # nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        if self.learn_resize:
            x = self.resize_img(x)
        x = self.backbone(x)
        # print(x.shape)
        x = self.fc(x)
        return x

    # def extract_feat(self, x):
    #     batch_size = x.shape[0]
    #     x = self.backbone(x)

    #     # if self.model_type != 'vit':
    #     #     x = self.pooling(x).view(batch_size, -1)
    #     x = self.dropout(x)
    #     x = self.fc(x)
    #     x = self.bn(x)
    #     x = self.fc_(x)
    #     return x
    
    def resize_img(self, x):
        res1 = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear')
        x = self.block1(x)
        res2 = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear')

        x = self.block2(res2)
        x += res2
        if self.r > 1:
            for _ in range(self.r):
                res2 = x
                x = self.block2(x)
                x += res2
 
        x = self.block3(x)
        x += res2
        
        x = self.block4(x)
        x += res1
        return x

def mixup_data(x, y, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_func(train_loader, model, device, criterion, optimizer, config):
    model.train()
    bar = tqdm(train_loader)
    if config["apex"]:
        scaler = GradScaler()
    losses = []
    for batch_idx, (images, targets) in enumerate(bar):
        images, targets = images.to(device, dtype=torch.float), targets.to(device, dtype=torch.float)
        #images, targets = images.cuda(), targets.cuda()
        if config["mixup"]:
            images, targets_a, targets_b, lam = mixup_data(images, targets.view(-1, 1), device)
            targets_a, targets_b = targets_a.to(device, dtype=torch.float), targets_a.to(device, dtype=torch.float)

        if config["debug"] and batch_idx == 100:
            print('Debug Mode. Only train on first 100 batches.')
            break

        # SAM
        if config["optimizer"] == "SAM":
            logits = model(images)
            # targets = targets.view(-1, 1)
            if config["mixup"]:
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                loss = criterion(logits, targets.view(-1, 1))
            loss.backward()
            optimizer.first_step(zero_grad=True)
            logits = model(images)
            if config["mixup"]:
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                loss = criterion(logits, targets.view(-1, 1))
            loss.backward()
            optimizer.second_step(zero_grad=True)
        else:
            if config["apex"]:
                with autocast():
                    logits = model(images)
                    if config["mixup"]:
                        loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                    else:
                        loss = criterion(logits, targets.view(-1, 1))
            else:
                logits = model(images)
                if config["mixup"]:
                    loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                else:
                    loss = criterion(logits, targets.view(-1, 1))
            
            if config["apex"]:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])

            if config["apex"]:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.item())
        smooth_loss = np.mean(losses[-30:])

        bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}')

    loss_train = np.mean(losses)
    return loss_train

def valid_func(train_loader, model, device, criterion, config):
    model.eval()
    bar = tqdm(train_loader)

    TARGETS = []
    losses = []
    PREDS = []
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(bar):
            images, targets = images.to(device, dtype=torch.float), targets.to(device, dtype=torch.float)
            #images, targets = images.cuda(), targets.cuda()
            if config["debug"] and batch_idx == 100:
                print('Debug Mode. Only train on first 100 batches.')
                break

            logits = model(images)

            # ??????????????????
            # PREDS += [torch.argmax(logits, 1).detach().cpu()]

            # ????????????
            PREDS += [logits.detach().cpu()]
            TARGETS += [targets.detach().cpu()]

            targets = torch.unsqueeze(targets, 1).type_as(logits)
            loss = criterion(logits, targets)
            losses.append(loss.item())
            smooth_loss = np.mean(losses[-30:])

            bar.set_description(f'loss: {loss.item():.5f}')

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    accuracy = roc_auc_score(TARGETS, PREDS)

    loss_valid = np.mean(losses)
    return loss_valid, accuracy

def get_prediction(model, valid_loader, device):
    preds = []
    sig=torch.nn.Sigmoid()
    with torch.no_grad():
        for img,label in tqdm(valid_loader):
            #img = img.cuda()
            img = img.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.float)
            feat = model(img)
            # feat = sig(feat)[..., 0]
            image_prediction = feat.detach().cpu().numpy()
            preds.append(image_prediction)
    image_predictions = np.concatenate(preds)
    image_predictions = np.array(image_predictions)
    image_predictions = sig(torch.from_numpy(image_predictions))
    return image_predictions

def get_image_embeddings(model, valid_loader, device):
    embeds = []

    with torch.no_grad():
        for img,label in tqdm(valid_loader):
            #img = img.cuda()
            img = img.to(device)
            label = label.to(device).long()
            #label = label.cuda()
            feat = model(img,label)
            image_embeddings = feat.detach().cpu().numpy()
            embeds.append(image_embeddings)

    del model
    image_embeddings = np.concatenate(embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    del embeds
    gc.collect()

    return image_embeddings

class MyScheduler(_LRScheduler):
    def __init__(self, optimizer, lr_start=5e-6, lr_max=1e-5,
                 lr_min=1e-6, lr_ramp_ep=5, lr_sus_ep=0, lr_decay=0.8,
                 last_epoch=-1):
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_ramp_ep = lr_ramp_ep
        self.lr_sus_ep = lr_sus_ep
        self.lr_decay = lr_decay
        super(MyScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            self.last_epoch += 1
            return [self.lr_start for _ in self.optimizer.param_groups]

        lr = self._compute_lr_from_epoch()
        self.last_epoch += 1

        return [lr for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return self.base_lrs

    def _compute_lr_from_epoch(self):
        if self.last_epoch < self.lr_ramp_ep:
            lr = ((self.lr_max - self.lr_start) /
                  self.lr_ramp_ep * self.last_epoch +
                  self.lr_start)

        elif self.last_epoch < self.lr_ramp_ep + self.lr_sus_ep:
            lr = self.lr_max

        else:
            lr = ((self.lr_max - self.lr_min) * self.lr_decay**
                  (self.last_epoch - self.lr_ramp_ep - self.lr_sus_ep) +
                  self.lr_min)
        return lr

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    '''
    T_0 : ??????????????????????????????
    T_mult : ?????????????????????????????????
    '''
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, eta_min=1e-3, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
        self.base_lrs = [eta_min]
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr