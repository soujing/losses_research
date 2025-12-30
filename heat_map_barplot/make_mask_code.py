#!/usr/bin/env python
# coding: utf-8

# # 이 코드는 같은 연구실에서 연구중인 안혜성님과 김승현님의 도움을 받았습니다.
# * Thanks To Hye-seong An
# * Thanks To Seung-hyun Kim 

# # 라이브러리로드

# In[1]:


import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

import mmcv
import torch

print(sys.path)

# from model.config import DEVICE
torch.cuda.set_device(0)  # 메인 파일에서 GPU 2번을 기본 디바이스로 설정
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Model is using: {torch.cuda.current_device()}")

# files and system
import sys
import time
import random
import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
# working with images
import cv2
import imageio
import scipy.ndimage
# import skimage.transform

import torchvision.transforms as transforms

# import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from tqdm import notebook

sys.path.insert(0, '..')

# losses
from metrics_loss import *

import pickle


# # 함수
# - 정확도 측정 함수 (iou, dice, precision, recall, f1)
# - 데이터 로드
# - 모델 로드
# - 손실함수 로드
# - 모델 학습 및 이미지 선정

# In[2]:


def eval_segmentation(outputs: torch.Tensor, labels: torch.Tensor, metric, batch_output=False):
    # outputs가 dict이거나 tuple일 경우, tensor를 가져옴
    if isinstance(outputs, dict):
        outputs = outputs['out']
    elif isinstance(outputs, tuple):
        outputs = outputs[0]

    # sigmoid를 적용하여 예측값을 확률(0~1)로 변환
    # 만약, 모델에 sigmoid나 이에 상응하는 활성화 함수가 포함되어 있으면 아래 줄을 주석 처리할 것
    outputs = torch.sigmoid(outputs)

    # 픽셀 별 예측 값을 0.5를 기준으로 0 또는 1로 thresholding
    outputs = outputs > 0.5

    # binary class의 경우 출력 channel은 1이므로, (BATCH, 1, H, W)의 형식을 가짐
    # 따라서, (BATCH, 1, H, W) -> (BATCH, H, W)로 차원을 줄여줌
    outputs = outputs.squeeze(1).byte()  # (BATCH, 1, H, W) -> (BATCH, H, W)
    labels = labels.squeeze(1).byte()    # (BATCH, 1, H, W) -> (BATCH, H, W)

    # SMOOTH는 나눗셈에서 분모가 0인 것을 방지하기 위해 더해주는 값
    SMOOTH = 1e-8

    if metric == 'iou':
        # IoU : intersection / union
        intersection = (outputs & labels).float().sum((1, 2))  # (BATCH, H, W)에서 픽셀 단위로 AND 연산 후 합산
        union = (outputs | labels).float().sum((1, 2))         # (BATCH, H, W)에서 픽셀 단위로 OR 연산 후 합산
        result = (intersection + SMOOTH) / (union + SMOOTH)
    elif metric == 'dice':
        # Dice Coefficient: 2 * intersection / (output + label)
        intersection = (outputs & labels).float().sum((1, 2))  # (BATCH, H, W)에서 픽셀 단위로 AND 연산 후 합산
        result = (2 * intersection + SMOOTH) / (outputs.float().sum((1, 2)) + labels.float().sum((1, 2)) + SMOOTH)
    elif metric == 'precision':
        # Precision: TP / (TP + FP)
        true_positive = (outputs & labels).float().sum((1, 2))  # True Positive (TP): 예측과 실제가 모두 1인 경우
        predicted_positive = outputs.float().sum((1, 2))        # Predicted Positive: 예측이 1인 경우
        result = (true_positive + SMOOTH) / (predicted_positive + SMOOTH)
    elif metric == 'recall':
        # Recall: TP / (TP + FN)
        true_positive = (outputs & labels).float().sum((1, 2))  # True Positive (TP): 예측과 실제가 모두 1인 경우
        actual_positive = labels.float().sum((1, 2))            # Actual Positive: 실제가 1인 경우
        result = (true_positive + SMOOTH) / (actual_positive + SMOOTH)
    elif metric == 'f1':
        # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        true_positive = (outputs & labels).float().sum((1, 2))  # True Positive (TP): 예측과 실제가 모두 1인 경우
        predicted_positive = outputs.float().sum((1, 2))        # Predicted Positive: 예측이 1인 경우
        actual_positive = labels.float().sum((1, 2))            # Actual Positive: 실제가 1인 경우
        # Precision과 Recall 계산
        precision = (true_positive + SMOOTH) / (predicted_positive + SMOOTH)
        recall = (true_positive + SMOOTH) / (actual_positive + SMOOTH)
        # F1 Score 계산
        result = (2 * precision * recall) / (precision + recall + SMOOTH)  

    if batch_output:
        return result  # shape: [BATCH] : 배치 내 각 이미지별 값 (벡터)
    else:
        return result.mean()  # shape: float (단일 상수 값) : 배치 내 모든 이미지의 평균 (상수)


# # 데이터 & model load

# In[3]:


def need_load_data(data_name, split_ratio, random_seed):
    df = pd.read_csv('./ALL_DATA.csv')
    image_files = list(df[df['type'] == data_name]['images'])
    label_files = list(df[df['type'] == data_name]['labels'])

    if len(split_ratio) == 2:  # split_ratio = (train, test)
        test_size = split_ratio[1] / sum(split_ratio)
        train_images, test_images, train_labels, test_labels = train_test_split(image_files, label_files, test_size=test_size, random_state=random_seed)
        return train_images, test_images, train_labels, test_labels
        
    elif len(split_ratio) == 3:  # split_ratio = (train, validation, test)
        trainval2test_size = split_ratio[2] / sum(split_ratio)
        trainval_images, test_images, trainval_labels, test_labels = train_test_split(image_files, label_files, test_size=trainval2test_size, random_state=random_seed)
        train2val_size = split_ratio[1] / sum(split_ratio[0:2])
        train_images, val_images, train_labels, val_labels = train_test_split(trainval_images, trainval_labels, test_size=train2val_size, random_state=random_seed)
        return train_images, val_images, test_images, train_labels, val_labels, test_labels
    else:
        img = df[df['type'] == data_name]['images']
        lab = df[df['type'] == data_name]['labels']
        return _, _, img, _, _, lab
        

def init_model(model_name):
    
    model = None  # 기본값을 None으로 설정하여 변수가 초기화되지 않는 상황 방지
    
    def get_project_root():
        try:
            # .py 파일에서 실행될 때
            return os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # Jupyter Notebook에서 실행될 때
            return os.getcwd()

    
    sys.path.insert(
    0,
    "/project/ahnailab/jsj0414/losses_research/model"
    )
    
    if model_name == 'FCBFormer':
        from FCBformer.FCBmodels import FCBFormer
        model = FCBFormer(size=224)
        
    # 모델이 None인 경우, 예외 처리
    if model is None:
        raise ValueError(f"모델 이름 '{model_name}'이 잘못되었거나 모델을 로드할 수 없습니다.")
    
    return model


models = ['FCBFormer'] # ['ColonFormer','DuckNet','UNet++','Unet','ESFPNet','DeepLab_V3+','FCN','ColonSegNet', 'EMCADNet','FCBFormer','caranet','FAT_Net']  # , 'EMCADNet','FCBFormer','caranet','FAT_Net'
for model in models:
    try: 
        init_model(model)
        print(f'{model} load가능\n')
    except Exception as e:
        print(f'{model} 로드 중 에러발생, 모델 로드 불가: {e}\n')


# In[21]:


def load_loss(loss_name):
    dic_name2func={
        "IoUWithGaussianLoss" : IoUWithGaussianLoss,
        "BCEWithGaussianLoss": BCEWithGaussianLoss,
        "DiceWithGaussianLoss": DiceWithGaussianLoss,
        "FocalWithGaussianLoss" : FocalWithGaussianLoss,
        
        "FocalWithSSIMLoss": FocalWithSSIMLoss,
        "IoUWithSSIMLoss" : IoUWithSSIMLoss,
        "DiceWithSSIMLoss" : DiceWithSSIMLoss,
        "BCEWithSSIMLoss" : BCEWithSSIMLoss,
        
        "FocalWithContextualLoss": FocalWithContextualLoss,
        "BCEWithContextualLoss": BCEWithContextualLoss,
        "IoUWithContextualLoss" : IoUWithContextualLoss,
        "DiceWithContextualLoss" : DiceWithContextualLoss,
        
        "IoUWithTVLoss_tar": IoUWithTVLoss_tar,
        "FocalWithTVLoss_tar" : FocalWithTVLoss_tar,
        "DiceWithTVLoss_tar" : DiceWithTVLoss_tar,
        "BCEWithTVLoss_tar" : BCEWithTVLoss_tar,
        
        
        "FocalWithEdgeLoss": FocalWithEdgeLoss,
        "BCEWithEdgeLoss": BCEWithEdgeLoss,
        "IoUWithEdgeLoss" : IoUWithEdgeLoss,
        "DiceWithEdgeLoss" : DiceWithEdgeLoss,
        
    
        'IoULoss' : IoULoss,
        'DiceLoss' : DiceLoss,
        'BCELoss' : BCELoss,
        'FocalLoss' : FocalLoss,
        'IoUDiceLoss' : IoUDiceLoss,
        'IoUBCELoss' : IoUBCELoss,
        'IoUFocalLoss' : IoUFocalLoss,
        'DiceBCELoss' : DiceBCELoss,
        'DiceFocalLoss' : DiceFocalLoss,
        'BCEFocalLoss' : BCEFocalLoss,
    }
    return dic_name2func[loss_name]()


# # training_model

# In[12]:


def training_model(data_names, model_name, loss_names, split_ratio=[0.6, 0.2, 0.2], base_random_seed=42, epochs=200, patience=50, BATCH_SIZE=8):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for n_data, data_name in enumerate(data_names):
        if base_random_seed != None:
            random_seed = base_random_seed
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            torch.backends.cudnn.deterministic = True

        train_images, val_images, test_images, train_labels, val_labels, test_labels = need_load_data(data_name, split_ratio=split_ratio, random_seed=random_seed)

        custom_dataset_train = myDataSet(train_images, train_labels, transforms=test_transforms)
        custom_dataset_val = myDataSet(val_images, val_labels, transforms=test_transforms)
        custom_dataset_test = myDataSet(test_images, test_labels, transforms=test_transforms)
       
        dataloader_train = torch.utils.data.DataLoader(custom_dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        dataloader_val = torch.utils.data.DataLoader(custom_dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        dataloader_test = torch.utils.data.DataLoader(custom_dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        model = init_model(model_name)
        model = model.to(DEVICE)

        for n_loss, loss_name in enumerate(loss_names):
            print(f"Start the task training_model : {n_data+1}'s data is {data_name} and {n_loss+1}'s loss function is {loss_name}")
            criterion = load_loss(loss_name)
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-8)                    

            state = {'best_val_dice' : 0, 'best_val_loss' : np.inf, 'best_epoch' : 0, 'last_epoch' : -1}    
        
            for epoch in range(epochs):
                model.train()
                for imgs, masks in dataloader_train:
                    if len(imgs) == 1: continue
                        
                    imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                    prediction = model(imgs)

                    if isinstance(prediction, dict):
                        prediction = torch.Tensor(prediction['out'])
                    elif isinstance(prediction, tuple):
                        prediction = torch.Tensor(prediction[0])
                    else:
                        prediction = prediction     
                
                    optimizer.zero_grad()
                    loss = criterion(prediction, masks)
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_loss, val_num, val_iou, val_dice, val_f1 = 0, 0, 0, 0, 0
                with torch.no_grad():
                    for imgs, masks in dataloader_val:
                        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                        prediction = model(imgs)

                        if isinstance(prediction, dict):
                            prediction = torch.Tensor(prediction['out'])
                        elif isinstance(prediction, tuple):
                            prediction = torch.Tensor(prediction[0])
                        else:
                            prediction = prediction    
                
                        loss = criterion(prediction, masks)
                        val_loss += loss.item()
                        val_num += len(imgs)
                        val_iou += eval_segmentation(prediction, masks, metric='iou', batch_output=True).sum()
                        val_dice += eval_segmentation(prediction, masks, metric='dice', batch_output=True).sum()
                        val_f1 += eval_segmentation(prediction, masks, metric='f1', batch_output=True).sum()

                # compute epoch-overall metric for val        
                epoch_val_loss = val_loss/len(dataloader_val)
                epoch_val_iou = (val_iou/val_num).item()
                epoch_val_dice = (val_dice/val_num).item()
                epoch_val_f1 = (val_f1/val_num).item()
            
                if epoch_val_dice >= state['best_val_dice']:
                    state['best_val_dice'] = epoch_val_dice
                    print(f'\tSaving.. {epoch+1} of {epochs}, best_val_dice improved from {state['best_val_dice']:.4f} to {epoch_val_dice:.4f}')
                    
                    # /project/ahnailab/jsj0414/지역포함_0113/working_path/model_recorder/{data_name}_{loss_name}.pth
                    torch.save(model.state_dict(),f"./model_recorder_alls/{data_name}_{loss_name}.pth")
            
                elif epoch - state['best_epoch'] > patience:
                    print(f"\n\tEarly stopping. Target criteria has not improved for {patience} epochs.\n")
                    break


# # saving the predict mask results

# In[13]:


def we_want_to_do(data_names, model_name, loss_names, split_ratio=[0.6,0.2,0.2], base_random_seed=42, epochs=200, patience=50, BATCH_SIZE=8):

    result = {}

    check_all_black_mask= []
    
    model = init_model(model_name)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for data_name in data_names:
        result[data_name] = {}
        score_history = {}
        loss_history = {}

        if base_random_seed != None:
            random_seed = base_random_seed
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            torch.backends.cudnn.deterministic = True

        _, _, test_images, _, _, test_labels = need_load_data(data_name, split_ratio=split_ratio, random_seed=random_seed)
        custom_dataset_test = myDataSet(test_images, test_labels, transforms=test_transforms)
        dataloader_test = torch.utils.data.DataLoader(custom_dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        test_img_cnt = len(dataloader_test)

        for loss_name in loss_names:
                                            # /project/ahnailab/jsj0414/지역포함_0113/working_path/model_recorder/{data_name}_{loss_name}.pth", weights_only=True
            print(f"\n ./model_recorder_alls/{data_name}_{loss_name}.pth 통과 전\n")
            model.load_state_dict(torch.load(f"./model_recorder_alls/{data_name}_{loss_name}.pth", weights_only=True))
            print(f"\n./model_recorder_alls/{data_name}_{loss_name}.pth 통과 \n")
            model.to(DEVICE)

            criterion = load_loss(loss_name)
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-8) 
        
            model.eval()
            with torch.no_grad():
                for i, (imgs, masks) in enumerate(dataloader_test):
                    # 라벨이 없는 데이터가 존재. 이를 후보에서 제외.
                    if masks.min() == masks.max():
                        test_img_cnt -= 1
                        with open("./model_recorder_alls/all_black_or_white_numb.txt", "a") as f:
                            f.write(f"{data_name} : {i}\n")
                        print(f"\n data : {data_name} img_numb : {i} \n")
                        continue
                    
                    imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                    prediction = model(imgs)
        
                    if isinstance(prediction, dict):
                        prediction = torch.Tensor(prediction['out'])
                    elif isinstance(prediction, tuple):
                        prediction = torch.Tensor(prediction[0])
                    else:
                        prediction = prediction    
                
                    loss = criterion(prediction, masks)
                    test_loss = loss.item()
                    test_f1 = eval_segmentation(prediction, masks, metric='f1', batch_output=True).sum().item()
        
                    img_score = test_f1
                    
                    if not i in loss_history:
                        loss_history[i] = [imgs.cpu(), masks.cpu()]
                    loss_history[i].append((loss_name, prediction))
                    # 
                    if not i in score_history:
                        score_history[i] = 0
                    score_history[i] += img_score

                    del imgs, masks, prediction, test_f1, img_score
                    torch.cuda.empty_cache()
            
        # print(f'{data_name} : number of the test image is {test_img_cnt}')

        score_lst = np.array(list(score_history.values()))

        mid_res = {}
        for _ in range(len(score_lst)):
            pick_numb = np.argmax(score_lst)
            
            key_list = list(loss_history.keys())
            img_numb = key_list[pick_numb]
            
            try:
                mid_res[img_numb] = [loss_history[pick_numb]] 
                score_lst[pick_numb] = 0
            
            except KeyError:
                with open("./making_mask/missing_keys_VA.txt", "a") as f:
                    f.write(f"{data_name} : {img_numb}\n")
                    score_lst[pick_numb] = 0
                continue
            
        
        result[data_name] = mid_res

    return result


# # check the re-train path

# In[24]:


def we_want_to_check(data_names, model_name, loss_names):

    have_to_retry_data_name = []
    have_to_retry_loss_name = []
    
    model = init_model(model_name)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for data_name in data_names:
        for loss_name in loss_names:
            try:
                # /project/ahnailab/jsj0414/지역포함_0113/working_path/model_recorder/{data_name}_{loss_name}.pth", weights_only=True
                print(f"\n ./model_recorder_alls/{data_name}_{loss_name}.pth 통과 전\n")
                model.load_state_dict(torch.load(f"./model_recorder_alls/{data_name}_{loss_name}.pth", weights_only=True))
                print(f"\n./model_recorder_alls/{data_name}_{loss_name}.pth 통과 \n")
            except : 
                print(f"\n 재학습 필요 :  ./model_recorder_alls/{data_name}_{loss_name}.pth\n")
                have_to_retry_data_name.append(f"{data_name}")
                have_to_retry_loss_name.append(f"{loss_name}") 
    have_to_retry_data_name = np.unique(have_to_retry_data_name)
    have_to_retry_loss_name = np.unique(have_to_retry_loss_name)
    
    return have_to_retry_data_name, have_to_retry_loss_name


# # 데이터셋 클래스 생성
# > 해당 클래스는 이용하려는 이미지와 라벨의 모든 경로(/data/segmentation/...)의 리스트를 인자로 받는다.   

# In[14]:


_size = 224, 224
resize = transforms.Resize(_size, interpolation=0)

# set your transforms 
train_transforms = transforms.Compose([
                           transforms.Resize(_size, interpolation=0),
                           transforms.RandomRotation(180),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomCrop(_size, padding = 10), # needed after rotation (with original size)
                       ])

test_transforms = transforms.Compose([
                           transforms.Resize(_size, interpolation=0),
                       ])

# Save images to folder and create a custom dataloader that loads them from their path. More involved than method 1 but allows for greater flexibility
# Requires 3 functions: __init__ to initialize the object, and __len__ and __get__item for pytorch purposes. More functions can be added as needed, but those 3 are necessary for it to function with pytorch
class myDataSet(object):

    def __init__(self, path_images, path_masks, transforms):
        "Initialization"
        self.all_path_images = sorted(path_images)
        self.all_path_masks = sorted(path_masks)
        self.transforms = transforms

    def __len__(self):
        "Returns length of dataset"
        return len(self.all_path_images)  

    def __getitem__(self, index):
        "Return next item of dataset"
        
        if torch.is_tensor(index):        # 인덱스가 tensor 형태일 수 있으니 리스트 형태로 바꿔준다.
            index = index.tolist()
        
        # Define path to current image and corresponding mask
        path_img = self.all_path_images[index]
        path_mask = self.all_path_masks[index]

        # Load image and mask:
        #     .jpeg has 3 channels, channels recorded last
        #     .jpeg records values as intensities from 0 to 255
        #     masks for some reason have values different to 0 or 255: 0, 1, 2, 3, 4, 5, 6, 7, 248, 249, 250, 251, 252, 253, 254, 255
        img_bgr = cv2.imread(path_img) 
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # cv2는 채널이 BGR로 저장된다 -> 출력할 때 RGB로 바꿔줘야함
        img = img / 255  # 픽셀 값들을 0~1로 변환한다
        
        mask = cv2.imread(path_mask)[:, :, 0] / 255  # 마스크의 채널은 1개만 있으면 된다
        mask = mask.round() # binarize to 0 or 1 (이진분류)
        
        # note, resizing happens inside transforms
        
        # convert to Tensors and fix the dimentions
        img = torch.FloatTensor(np.transpose(img, [2, 0 ,1])) # Pytorch uses the channels in the first dimension
        mask = torch.FloatTensor(mask).unsqueeze(0) # Adding channel dimension to label
        
        # apply transforms/augmentation to both image and mask together
        sample = torch.cat((img, mask), 0) # insures that the same transform is applied
        sample = self.transforms(sample)
        img = sample[:img.shape[0], ...]
        mask = sample[img.shape[0]:, ...]

        return img, mask


# # 모델 학습과 이미지 선정

# In[15]:


alls = pd.read_csv('./ALL_DATA.csv')

all_data = alls["type"].unique().tolist() # 

all_models = 'FCBFormer'

all_losses = ["IoUWithContextualLoss","DiceWithContextualLoss","IoUFocalLoss","DiceBCELoss","IoULoss","DiceLoss"]


# # 학습 후 path 추가
# * 학습 후 주석처리하길 추천드립니다.

# In[16]:

loss = all_losses

data =alls["type"].unique().tolist()

data_name = ['wound', 'CVC-ClinicDB', 'Kvasir-SEG', 
            'breast-cancer-benign', 'breast-cancer-malignant', 'ISIC']

parameter = {
    'data_names' : data_name, 'model_name' : all_models, 'loss_names' : loss,
    'split_ratio' : [0.6, 0.2, 0.2], 'base_random_seed' : 42, 'epochs' : 200, 'patience' : 40, 'BATCH_SIZE' : 1
    # ,'pass_num' : None
}


# # pth load제대로 되나 확인하기

# In[29]:


parameter = {
    'data_names' : data_name,
    'model_name' : all_models,
    'loss_names' : all_losses
}

have_to_retry_data_name, have_to_retry_loss_name = we_want_to_check(**parameter)
print(have_to_retry_data_name,"\n")
print(have_to_retry_loss_name,"\n")


# # trainning

parameter = {
    'data_names' : have_to_retry_data_name,
    'model_name' : all_models,
    'loss_names' : have_to_retry_loss_name,
    'split_ratio' : [0.6, 0.2, 0.2], 
    'base_random_seed' : 42,
    'epochs' : 200,
    'patience' : 40,
    'BATCH_SIZE' : 8
}

training_model(**parameter)

# # saving mask predict results

parameter = {
    'data_names' : have_to_retry_data_name,
    'model_name' : all_models,
    'loss_names' : have_to_retry_loss_name,
    'split_ratio' : [0.6, 0.2, 0.2], 
    'base_random_seed' : 42,
    'epochs' : 200,
    'patience' : 40,
    'BATCH_SIZE' : 8
}

we_want_to_do(**parameter)


# # **그리기 함수**
# - 예측 이미지에 시그모이드 적용
# - 시그모이드 적용에 이진 분류 적용

# In[31]:


from torchvision.transforms.functional import resize as T_resize

def draw(data_name, result, n_row, row_lim=(0, -1), col_lim=(0, -1), title=False, great_numb=0):
    global axs

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    row_min, row_max = row_lim
    col_min, col_max = col_lim
# 
    datas = result[data_name][great_numb]
    max_cols = axs.shape[1]
    i = 0

    for data in datas:
        if isinstance(data, int):
            continue
        for d in data:
            if i >= max_cols:
                print(f"Reached max cols ({max_cols}), stopping.")
                return

            ax = axs[n_row, i]

            # 0: 원본 이미지 (3채널)
            if i == 0:
                img = d.squeeze(0).permute(1,2,0)  # (H, W, 3)
                img = T_resize(torch.tensor(img).permute(2, 0, 1), [224, 224])  # (3, H, W)
                img = img.permute(1, 2, 0).numpy()  # 다시 (H, W, 3)로 복원
                ax.imshow(img)

            # 1: 마스크(1채널 → 3채널 보기용)
            elif i == 1:
                # 원본 1채널 마스크를 저장해둠 (H,W)
                mask_tensor = d.squeeze(0)[0]      # (1,H,W) → (H,W)
                
                # resize로 224x224 맞추기 (crop 대신)
                mask_tensor = T_resize(mask_tensor.unsqueeze(0), [224, 224]).squeeze(0)
            
                # 보기 편하게 3채널로 변환
                mask_rgb = np.stack([mask_tensor.numpy()] * 3, axis=-1)
                
                ax.imshow(mask_rgb, cmap='gray')
                ax.axis('off')
                if title: ax.set_title("Mask", fontsize=12)

            # 2 이상: prediction → confusion map
            else:
                loss, pred = d
                loss_name = loss.replace('Loss','')
                
                if "Contextual" in loss_name:
                    loss_name = loss_name.split("With")[0]+"_Percepture"
                else:
                    loss_name = loss_name
                    
                # 1) 분산 맵 & 이진화
                var_map = pred
                cut_line = 0.5
                binary_pred = (var_map.squeeze() > cut_line).float()
                # 동일하게 crop
                binary_pred = T_resize(binary_pred.unsqueeze(0), [224, 224]).squeeze(0)

                # 2) label binary
                label_binary = mask_tensor.to(DEVICE) 

                # 3) confusion image 생성
                h, w = binary_pred.shape
                confusion = np.zeros((h, w, 3), dtype=float)
                tp = ((binary_pred==1)&(label_binary==1)).cpu().numpy()
                tn = ((binary_pred==0)&(label_binary==0)).cpu().numpy()
                fp = ((binary_pred==1)&(label_binary==0)).cpu().numpy()
                fn = ((binary_pred==0)&(label_binary==1)).cpu().numpy()
                confusion[tp] = [1,1,1]  # white
                confusion[fp] = [1,0,0]  # red
                confusion[fn] = [0,0,1]  # blue
                # tn는 검정(0,0,0)

                ax.imshow(confusion)
                ax.axis('off')
                if title:
                    ax.set_title(loss_name, fontsize=15)

            i += 1


# # **결과**
# > 10/20/35 pass

# In[28]:


need_parameter = {
    'data_names' : all_data, 'model_name' : all_models, 'loss_names' : all_losses,
    'split_ratio' : [1.0], 'base_random_seed' : 42, 'epochs' : 200, 'patience' : 40, 'BATCH_SIZE' : 1
    # ,'pass_num' : None
}

need_result = we_want_to_do(**need_parameter)


# # 저장된 결과 불러오기

# In[30]:


# # 파일로 저장
with open('./making_mask/all_result_f1_real_last.pkl', 'wb') as f:
    pickle.dump(need_result, f)


# In[32]:


with open('./making_mask/all_result_f1_real_last.pkl', 'rb') as f:
    loaded_dict_real_last = pickle.load(f)


# ## (완료)wound

# In[ ]:


all_keys = list(loaded_dict_real_last["wound"].keys())
key_all_len = len(all_keys)
for i in range(key_all_len):
    key = all_keys[i]
    fig, axs = plt.subplots(1, 8, figsize=(20, 5), squeeze=False)

    great_num=key
    
    _ = plt.suptitle(f'the pass_num is {i}', fontsize=30,ㅇy=0.9)
    draw('wound', loaded_dict_real_last, n_row=0, row_lim=(25,105), col_lim=(0,80), title=True, great_numb=great_num)  # great_numb=이미지 키번호 넣어야한다.
    
    # ====== 간격 조절 ======
    plt.tight_layout(h_pad=0.1, w_pad=0.5)  # h_pad, w_pad로 간격 조절
    
    plt.subplots_adjust(top=0.9)  # 제목이랑 subplot 사이 여백 조정
    plt.savefig(f"./making_mask/wound/F1_{i}_img_pred.png", bbox_inches='tight')


# ## CVC-ClinicDB

# In[ ]:


all_keys = list(loaded_dict_real_last["CVC-ClinicDB"].keys())
key_all_len = len(all_keys)
for i in range(key_all_len):
    key = all_keys[i]
    fig, axs = plt.subplots(1, 8, figsize=(20, 5), squeeze=False)

    great_num=key
    
    _ = plt.suptitle(f'the pass_num is {i}', fontsize=30, y=0.9)
    draw('CVC-ClinicDB', loaded_dict_real_last, n_row=0, row_lim=(25,105), col_lim=(0,80), title=True, great_numb=great_num)  
    
    # ====== 간격 조절 ======
    plt.tight_layout(h_pad=0.1, w_pad=0.5)  # h_pad, w_pad로 간격 조절
    
    plt.subplots_adjust(top=0.9)  # 제목이랑 subplot 사이 여백 조정
    plt.savefig(f"./making_mask/CVC/F1_{i}_img_pred.png", bbox_inches='tight')


# ## Kvasir

# In[ ]:


all_keys = list(loaded_dict_real_last["Kvasir-SEG"].keys())
key_all_len = len(all_keys)
for i in range(key_all_len):
    key = all_keys[i]
    fig, axs = plt.subplots(1, 8, figsize=(20, 5), squeeze=False)

    great_num=key
    
    _ = plt.suptitle(f'the pass_num is {i}', fontsize=30, y=0.9)
    draw('Kvasir-SEG', loaded_dict_real_last, n_row=0, row_lim=(25,105), col_lim=(0,80), title=True, great_numb=great_num) # great_numb=이미지 키번호 넣어야한다.
    
    # ====== 간격 조절 ======
    plt.tight_layout(h_pad=0.1, w_pad=0.5)  # h_pad, w_pad로 간격 조절
    
    plt.subplots_adjust(top=0.9)  # 제목이랑 subplot 사이 여백 조정
    plt.savefig(f"./making_mask/Kvasir/F1_{i}_img_pred.png", bbox_inches='tight')


# ## breast-cancer-benign

# In[ ]:


all_keys = list(loaded_dict_real_last["breast-cancer-benign"].keys())
key_all_len = len(all_keys)
for i in range(key_all_len):
    key = all_keys[i]
    fig, axs = plt.subplots(1, 8, figsize=(20, 5), squeeze=False)

    great_num=key
    
    _ = plt.suptitle(f'the pass_num is {i}', fontsize=30, y=0.9)
    draw('breast-cancer-benign', loaded_dict_real_last, n_row=0, row_lim=(25,105), col_lim=(0,80), title=True, great_numb=great_num)  # great_numb=이미지 키번호 넣어야한다.
    
    # ====== 간격 조절 ======
    plt.tight_layout(h_pad=0.1, w_pad=0.5)  # h_pad, w_pad로 간격 조절
    
    plt.subplots_adjust(top=0.9)  # 제목이랑 subplot 사이 여백 조정
    plt.savefig(f"./making_mask/begin/F1_{i}_img_pred.png", bbox_inches='tight')


# ## breast-cancer-malignant

# In[ ]:


all_keys = list(loaded_dict_real_last["breast-cancer-malignant"].keys())
key_all_len = len(all_keys)
for i in range(key_all_len):
    key = all_keys[i]
    fig, axs = plt.subplots(1, 8, figsize=(20, 5), squeeze=False)

    great_num=key
    
    _ = plt.suptitle(f'the pass_num is {i}', fontsize=30, y=0.9)
    draw('breast-cancer-malignant', loaded_dict_real_last, n_row=0, row_lim=(25,105), col_lim=(0,80), title=True, great_numb=great_num)  # great_numb=이미지 키번호 넣어야한다.
    
    # ====== 간격 조절 ======
    plt.tight_layout(h_pad=0.1, w_pad=0.5)  # h_pad, w_pad로 간격 조절
    
    plt.subplots_adjust(top=0.9)  # 제목이랑 subplot 사이 여백 조정
    plt.savefig(f"./making_mask/malignant/F1_{i}_img_pred.png", bbox_inches='tight')


# ## ISIC

# In[ ]:


all_keys = list(loaded_dict_real_last["ISIC"].keys())
key_all_len = len(all_keys)
for i in range(key_all_len):
    key = all_keys[i]
    fig, axs = plt.subplots(1, 8, figsize=(20, 5), squeeze=False)

    great_num=key
    
    _ = plt.suptitle(f'the pass_num is {i}', fontsize=30, y=0.9)
    draw('ISIC', loaded_dict_real_last, n_row=0, row_lim=(25,105), col_lim=(0,80), title=True, great_numb=great_num)  
    
    # ====== 간격 조절 ======
    plt.tight_layout(h_pad=0.1, w_pad=0.5)  # h_pad, w_pad로 간격 조절
    
    plt.subplots_adjust(top=0.9)  # 제목이랑 subplot 사이 여백 조정
    plt.savefig(f"./making_mask/ISIC/F1_{i}_img_pred.png", bbox_inches='tight')

