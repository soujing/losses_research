############################## 1. 현재 작업 환경 설정 변경 ########################
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)
import mmcv
import torch


# from model.config import DEVICE
torch.cuda.set_device(0)  # 메인 파일에서 GPU 0번을 기본 디바이스로 설정
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Model is using: {torch.cuda.current_device()}")


############################## 2. 라이브러리 로드 ########################
# files and system
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

import torchvision
import torch.nn as nn
import torch.nn.functional as F

from tqdm import notebook

# sys.path.insert(0, '..')

# losses
from metrics_loss import *

# model 전부 load
# import model


############################## 3. data load ########################
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


############################## 4. 데이터셋 클래스 생성 #################################################
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

############################## 5. Load data  ##############################
def load_data(data_name, split_ratio, random_seed):
    df = pd.read_csv('./ALL_DATA.csv')
    df = pd.DataFrame({col: np.array(df[col]) for col in df.columns})
    
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

############################ 6. 모델을 load ############################
def init_model(model_name):

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
    
    model = None  # 기본값을 None으로 설정하여 변수가 초기화되지 않는 상황 방지
    
    if model_name == 'FCBFormer':
        from FCBformer.FCBmodels import FCBFormer
        model = FCBFormer(size=224)
        
    elif model_name == 'EMCADNet':
        from EMCAD import EMCADNet
        model = EMCADNet(encoder='pvt_v2_b2')

    elif model_name == 'ColonSegNet':
        from ColonSegNet import CompNet as ColonSegNet
        model = ColonSegNet()

    elif model_name == 'FCN':
        from FCN.models.segmentation.fcn import fcn_resnet101
        model = fcn_resnet101(num_classes=1)

    elif model_name == 'DeepLab_V3+':
        from DeepLab_V3_p.model import DeepLab as DeepLab_V3_p
        model = DeepLab_V3_p(backbone='resnet', num_classes=1)

    elif model_name == 'ESFPNet':
        from ESFPNet.ESFPmodel import ESFPNetStructure
        model = ESFPNetStructure(embedding_dim=224)


    elif model_name == 'Unet':
        from Unet.unet import UNet
        model = UNet(n_channels=3, n_classes=1, pretrained=True)

    elif model_name == 'UNet++':
        from nnunet import Nested_UNet as UNet2p
        model = UNet2p(1, 3)

    elif model_name == 'DuckNet':
        from duck_net import DuckNet
        model = DuckNet(in_channels=3, out_channels=1, depth=5, init_features=34, normalization='batch', interpolation='nearest', out_activation=None, use_multiplier=True)
        import torch.nn as nn
        model.apply(lambda m: nn.init.kaiming_uniform_(m.weight) if type(m) == nn.Conv2d else None) # default init is xaiver uniform        

    elif model_name == 'ColonFormer':
        # 이 파일(init_model.py 또는 notebook이 아니라,
        # "이 코드를 포함한 파일" 기준)
        COLONFORMER_ROOT = "/project/ahnailab/jsj0414/losses_research/model/original_ColonFormer/"
        sys.path.append(os.path.abspath(COLONFORMER_ROOT))  
          
        from colon_lib.models.segmentors.colonformer import ColonFormer
        
        backbone=dict(type='mit_b3',style='pythorch')
        
        decode_head=dict(type='UPerHead', in_channels=[64], in_index=[0], channels=128, dropout_ratio=0.1,
                            num_classes=1, norm_cfg=dict(type='BN', requires_grad=True), align_corners=False,decoder_params=dict(embed_dim=768),
                            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
        
        pretrained = os.path.join(
            COLONFORMER_ROOT,
            "colon_lib",
            "pretrained",
            "mit_b3.pth"
        )
        
        model = ColonFormer(backbone,decode_head = decode_head,
                        neck=None,
                        auxiliary_head=None,
                        train_cfg=dict(),
                        test_cfg=dict(mode='whole'),
                        pretrained= pretrained)

    elif model_name == 'caranet':
        from caranet import caranet
        
        model = caranet()

    elif model_name == 'FAT_Net':
        from FATNet_model.FAT_Net import FAT_Net
        model = FAT_Net()

    # 모델이 None인 경우, 예외 처리
    if model is None:
        raise ValueError(f"모델 이름 '{model_name}'이 잘못되었거나 모델을 로드할 수 없습니다.")
    
    return model


############################ 7. 모델을 load 여부 확인 ############################
models = ['ColonFormer','DuckNet','UNet++','Unet' ,'ESFPNet','DeepLab_V3+','FCN'
         ,'ColonSegNet','EMCADNet','FCBFormer','FAT_Net','caranet'] # 'ColonFormer',
for model in models:
    try: 
        init_model(model)
        print(f'{model} load가능\n')
    except Exception as e:
        print(f'{model} 로드 중 에러발생, 모델 로드 불가: {e}\n')


############################ 8. load loss ############################
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

############################ 9. 모델 클래스 생성 ############################

def run(data_names,    # e.g. ['breast-cancer-benign']
        model_names,   # e.g. ['FCBFormer']
        loss_names,    # e.g. ['DiceBCELoss']
        iters = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        split_ratio = [0.6, 0.2, 0.2],
        base_random_seed = 42, 
        epochs = 200, 
        patience = 50, 
        BATCH_SIZE = 8,
        result_file = None
       ):
    DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu") # select device for training, i.e. gpu or cpu

    import itertools
    for data_name, model_name, loss_name, iter in itertools.product(data_names, model_names, loss_names, iters):
        if base_random_seed != None:
            random_seed = base_random_seed + iter
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            torch.backends.cudnn.deterministic = True

        # load data
        train_images, val_images, test_images, train_labels, val_labels, test_labels = load_data(data_name, split_ratio=split_ratio, random_seed=random_seed)

        custom_dataset_train = myDataSet(train_images, train_labels, transforms=test_transforms)
        custom_dataset_val = myDataSet(val_images, val_labels, transforms=test_transforms)
        custom_dataset_test = myDataSet(test_images, test_labels, transforms=test_transforms)
       
        dataloader_train = torch.utils.data.DataLoader(custom_dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        dataloader_val = torch.utils.data.DataLoader(custom_dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        dataloader_test = torch.utils.data.DataLoader(custom_dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        print(f'Experiment env - \n'+
              f'\t - data name : {data_name}\n' +
              f'\t - the  ratio of dataset : total/train/val/test = {sum(split_ratio)}/{split_ratio[0]}/{split_ratio[1]}/{split_ratio[2]}\n' +
              f'\t - the number of dataset : total/train/val/test = {len(custom_dataset_train)+len(custom_dataset_val)+len(custom_dataset_test)}/{len(custom_dataset_train)}/{len(custom_dataset_val)}/{len(custom_dataset_test)}\n' +
              f'\t - model name : {model_name}\n' +
              f'\t - loss name : {loss_name}\n' +
              f'\t - iter : {iter} of {iters}\n' +
              f'\t - random_seed : {random_seed}\n' +
              f'\t - epochs & patience & BATCH_SIZE & DEVICE: {epochs} & {patience} & {BATCH_SIZE} & {DEVICE}')
        
        # initiate model
        model = init_model(model_name)
        model = model.to(DEVICE)

        # load loss
        criterion = load_loss(loss_name)

        # https://github.com/JunMa11/SegLossOdyssey/blob/master/losses_pytorch/hausdorff.py
        
        # Define optimiser
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-8)                    

        # state to record result
        state = {
            'train_losses' : [], 'train_ious' : [], 'train_dices' : [], 'train_precisions' : [], 'train_recalls' : [], 'train_f1s' : [],
            'val_losses' : [], 'val_ious' : [], 'val_dices' : [], 'val_precisions' : [], 'val_recalls' : [], 'val_f1s' : [],
            'test_losses' : [], 'test_ious' : [], 'test_dices' : [], 'test_precisions' : [], 'test_recalls' : [], 'test_f1s' : [],
            'best_val_dice' : 0,
            'best_val_loss' : np.inf,
            'best_epoch' : 0,
            'best_net' : None,
            'last_epoch' : -1
        }    
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss, train_num, train_iou, train_dice, train_precision, train_recall, train_f1 = 0, 0, 0, 0, 0, 0, 0
            for i, (imgs, masks) in enumerate(dataloader_train):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                
                prediction = model(imgs)

                # dict형태로 데이터가 들어오는 경우가 있음 ######################################################################
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
                
                train_loss += loss.item()
                train_num += len(imgs)
                train_iou += eval_segmentation(prediction, masks, metric='iou', batch_output=True).sum()
                train_dice += eval_segmentation(prediction, masks, metric='dice', batch_output=True).sum()
                train_precision += eval_segmentation(prediction, masks, metric='precision', batch_output=True).sum()
                train_recall += eval_segmentation(prediction, masks, metric='recall', batch_output=True).sum()
                train_f1 += eval_segmentation(prediction, masks, metric='f1', batch_output=True).sum()
                print("\r Epoch: {} of {}, Iter.: {} of {}, Train Loss: {:.6f}, Train IoU: {:.6f}, Train Dice:  {:.6f}".format(epoch, epochs, i, len(dataloader_train), train_loss/(i+1), train_iou/train_num, train_dice/train_num), end="")

            # compute epoch-overall metric for train
            epoch_train_loss = train_loss/len(dataloader_train)
            epoch_train_iou = (train_iou/train_num).item()
            epoch_train_dice = (train_dice/train_num).item()
            epoch_train_precision = (train_precision/train_num).item()
            epoch_train_recall = (train_recall/train_num).item()
            epoch_train_f1 = (train_f1/train_num).item()
                
            # Validate
            model.eval()
            val_loss, val_num, val_iou, val_dice, val_precision, val_recall, val_f1 = 0, 0, 0, 0, 0, 0, 0
            for i, (imgs, masks) in enumerate(dataloader_val):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                
                if len(imgs) != BATCH_SIZE:
                    continue                
                    
                prediction = model(imgs)

                # dict형태로 데이터가 들어오는 경우가 있음 ######################################################################
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
                val_precision += eval_segmentation(prediction, masks, metric='precision', batch_output=True).sum()
                val_recall += eval_segmentation(prediction, masks, metric='recall', batch_output=True).sum()
                val_f1 += eval_segmentation(prediction, masks, metric='f1', batch_output=True).sum()

            # compute epoch-overall metric for val        
            epoch_val_loss = val_loss/len(dataloader_val)
            epoch_val_iou = (val_iou/val_num).item()
            epoch_val_dice = (val_dice/val_num).item()
            epoch_val_precision = (val_precision/val_num).item()
            epoch_val_recall = (val_recall/val_num).item()
            epoch_val_f1 = (val_f1/val_num).item()

            # Test
            model.eval()
            test_loss, test_num, test_iou, test_dice, test_precision, test_recall, test_f1 = 0, 0, 0, 0, 0, 0, 0
            for i, (imgs, masks) in enumerate(dataloader_test):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                
                if len(imgs) != BATCH_SIZE:
                    continue
                    
                prediction = model(imgs)

                # dict형태로 데이터가 들어오는 경우가 있음 ######################################################################
                if isinstance(prediction, dict):
                    prediction = torch.Tensor(prediction['out'])
        
                elif isinstance(prediction, tuple):
                    prediction = torch.Tensor(prediction[0])
                
                else:
                    prediction = prediction   
                
                
                loss = criterion(prediction, masks)
                    
                test_loss += loss.item()
                test_num += len(imgs)
                test_iou += eval_segmentation(prediction, masks, metric='iou', batch_output=True).sum()
                test_dice += eval_segmentation(prediction, masks, metric='dice', batch_output=True).sum()
                test_precision += eval_segmentation(prediction, masks, metric='precision', batch_output=True).sum()
                test_recall += eval_segmentation(prediction, masks, metric='recall', batch_output=True).sum()
                test_f1 += eval_segmentation(prediction, masks, metric='f1', batch_output=True).sum()
           
            # compute epoch-overall metric for test
            epoch_test_loss = test_loss/len(dataloader_test)
            epoch_test_iou = (test_iou/test_num).item()
            epoch_test_dice = (test_dice/test_num).item()
            epoch_test_precision = (test_precision/test_num).item()
            epoch_test_recall = (test_recall/test_num).item()
            epoch_test_f1 = (test_f1/test_num).item()

            # print out the result
            print(f"\r Epoch: {epoch} of {epochs}, "+
                  f"Loss: {epoch_train_loss:.4f}/{epoch_val_loss:.4f}/{epoch_test_loss:.4f}, "+
                  f"IoU: {epoch_train_iou:.4f}/{epoch_val_iou:.4f}/{epoch_test_iou:.4f}, "+
                  f"Dice: {epoch_train_dice:.4f}/{epoch_val_dice:.4f}/{epoch_test_dice:.4f}")
            
            # record the result
            state['train_losses'].append(epoch_train_loss)
            state['train_ious'].append(epoch_train_iou)
            state['train_dices'].append(epoch_train_dice)
            state['train_precisions'].append(epoch_train_precision)
            state['train_recalls'].append(epoch_train_recall)
            state['train_f1s'].append(epoch_train_f1)
            state['val_losses'].append(epoch_val_loss)
            state['val_ious'].append(epoch_val_iou)
            state['val_dices'].append(epoch_val_dice)
            state['val_precisions'].append(epoch_val_precision)
            state['val_recalls'].append(epoch_val_recall)
            state['val_f1s'].append(epoch_val_f1)            
            state['test_losses'].append(epoch_test_loss)
            state['test_ious'].append(epoch_test_iou)
            state['test_dices'].append(epoch_test_dice)
            state['test_precisions'].append(epoch_test_precision)
            state['test_recalls'].append(epoch_test_recall)
            state['test_f1s'].append(epoch_test_f1)                
            state['last_epoch'] = epoch
            
            if epoch_val_dice >= state['best_val_dice']:
                print(f'Saving.. {epoch} of {epochs}, best_val_dice improved from {state['best_val_dice']:.4f} to {epoch_val_dice:.4f}')
                state['best_val_dice'] = epoch_val_dice                                          
                state['best_epoch'] = epoch

                # state['best_net'] = model.state_dict()
                # if not os.path.isdir('checkpoints'):
                #     os.mkdir('checkpoints')
                # torch.save(state, f'./checkpoints/ckpt_{model_name}_{data_name}.pth')
            
            elif epoch - state['best_epoch'] > patience:
                print(f"\nEarly stopping. Target criteria has not improved for {patience} epochs.\n")
                break
        
        # print(f'Validationset 기준 \nBest_epoch:{best_epoch_dice}, Best_IOU:{best_iou:.4f}, Best_DiceScore:{best_dice:.4f}')
        
        fig, axs = plt.subplots(nrows=1, ncols=6, figsize=(18, 9))
        for i, metric in enumerate(['losses','ious','dices','precisions','recalls','f1s']):
            axs[i].plot(np.arange(state['last_epoch']+1), state['train_'+metric], label=f'Train, {metric}', linewidth=2, color='blue')
            axs[i].plot(np.arange(state['last_epoch']+1), state['val_'+metric], label=f'Val, {metric}', linewidth=2, color='green')
            axs[i].plot(np.arange(state['last_epoch']+1), state['test_'+metric], label=f'Test, {metric}', linewidth=2, color='orange')

            # best_epoch 시점에 수직선 추가
            axs[i].axvline(x=state['best_epoch'], color='r', linestyle='--', label=f'Best epoch : {state["best_epoch"]}')
            
            # best_epoch에서의 train, val, test 값을 텍스트로 표시
            best_train_value = state['train_' + metric][state['best_epoch']]
            best_val_value = state['val_' + metric][state['best_epoch']]
            best_test_value = state['test_' + metric][state['best_epoch']]
        
            # 각 점에 marker를 찍고 텍스트로 값 표시
            axs[i].scatter(state['best_epoch'], best_train_value, color='blue', zorder=5)
            axs[i].scatter(state['best_epoch'], best_val_value, color='green', zorder=5)
            axs[i].scatter(state['best_epoch'], best_test_value, color='orange', zorder=5)
        
            # 텍스트로 표시
            axs[i].annotate(f'Train: {best_train_value:.4f}', 
                            (state['best_epoch'], best_train_value), 
                            textcoords="offset points", xytext=(0,10), ha='center', color='blue')
            axs[i].annotate(f'Val: {best_val_value:.4f}', 
                            (state['best_epoch'], best_val_value), 
                            textcoords="offset points", xytext=(0,10), ha='center', color='green')
            axs[i].annotate(f'Test: {best_test_value:.4f}', 
                            (state['best_epoch'], best_test_value), 
                            textcoords="offset points", xytext=(0,10), ha='center', color='orange')

            axs[i].set_xlabel('Epoch')
            axs[i].set_ylabel(metric)
            axs[i].set_title(f'{metric}')
            axs[i].legend(loc='best')
        plt.show()
        
        if not os.path.isfile(result_file):
            fp = open(result_file, 'w')
            fp.write(','.join(['data_name',
                           'model_name',
                           'loss_name',
                           'iter',
                           'best_epoch',
                           'test_iou',
                           'test_dice',
                           'test_precision',
                           'test_recall',
                           'test_f1'])+'\n')
        else:
            fp = open(result_file, 'a')
        best_epoch = state['best_epoch']
        fp.write(','.join(map(str, [data_name,
                           model_name,
                           loss_name,
                           iter,
                           best_epoch,
                           state['test_ious'][best_epoch],
                           state['test_dices'][best_epoch],
                           state['test_precisions'][best_epoch],
                           state['test_recalls'][best_epoch],
                           state['test_f1s'][best_epoch]]))+'\n')
        fp.close()



############################ 10. 모든 loss ############################
all_losses = [
        "IoUWithGaussianLoss",
        "BCEWithGaussianLoss",
        "DiceWithGaussianLoss",
        "FocalWithGaussianLoss",
        
        "FocalWithSSIMLoss",
        "IoUWithSSIMLoss" ,
        "DiceWithSSIMLoss"  ,
        "BCEWithSSIMLoss"  ,
        
        "FocalWithContextualLoss" ,
        "BCEWithContextualLoss" ,
        "IoUWithContextualLoss"  ,
        "DiceWithContextualLoss"  ,
        
        "IoUWithTVLoss_tar" ,
        "FocalWithTVLoss_tar"  ,
        "DiceWithTVLoss_tar"  ,
        "BCEWithTVLoss_tar" ,
        
        
        "FocalWithEdgeLoss" ,
        "BCEWithEdgeLoss" ,
        "IoUWithEdgeLoss" ,
        "DiceWithEdgeLoss"  ,
        
    
        'IoULoss' ,
        'DiceLoss', 
        'BCELoss' ,
        'FocalLoss' ,
        'IoUDiceLoss' ,
        'IoUBCELoss' ,
        'IoUFocalLoss' ,
        'DiceBCELoss' ,
        'DiceFocalLoss' ,
        'BCEFocalLoss' ,
]


############################ 12. 데이터, 모델 그리고 iter 정리 ############################

# data names
data_names = ['CVC-ClinicDB', 'ISIC', 'Kvasir-SEG', 'breast-cancer-benign', 'breast-cancer-malignant', 'wound']

# models name
models = ['ColonFormer','FCN','DuckNet','UNet++','Unet','ESFPNet','DeepLab_V3+','ColonSegNet','EMCADNet','FCBFormer','caranet','FAT_Net'] 

iters = list(np.arange(0,5,1))

print(f'data_names: {data_names}')
print(f'models: {models}')
print(f'iters: {iters}')




# ############################ Example run ############################
# run_env = {
#     'data_names' : ['CVC-ClinicDB'],  # [str(data) for data in data_names],
#     'model_names' : ["ColonFormer"], # [str(model) for model in models],
#     'loss_names' : ["DiceWithSSIMLoss"], # [str(loss) for loss in all_losses],
#     'iters' : [0], # [int(it) for it in iters],
#     'split_ratio' : [0.6, 0.2, 0.2],
#     'base_random_seed' : 42,  
#     'epochs' : 200,
#     'patience' : 40,
#     'BATCH_SIZE' : 8,
#     'result_file' : './result.csv'
# }

# run(**run_env)


############################ 13. RUNNING ############################

run_env = {
    'data_names' :  [str(data) for data in data_names],
    'model_names' : [str(model) for model in models],
    'loss_names' : [str(loss) for loss in all_losses],
    'iters' : [int(it) for it in iters],
    'split_ratio' : [0.6, 0.2, 0.2],
    'base_random_seed' : 42,  
    'epochs' : 200,
    'patience' : 40,
    'BATCH_SIZE' : 8,
    'result_file' : './result.csv'
}

run(**run_env)