#!/usr/bin/env python
# coding: utf-8

# # 라이브러리 로드

# In[4]:


import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)


# # DATA_LOAD

# In[148]:


new_loss = [ "IoUWithGaussianLoss",
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
        "DiceWithEdgeLoss"]

pres_loss = ['IoULoss', 'DiceLoss', 'BCELoss', 'FocalLoss', 'IoUDiceLoss', 'IoUBCELoss', 'IoUFocalLoss', 
'DiceBCELoss', 'DiceFocalLoss', 'BCEFocalLoss']

all_loss = new_loss + pres_loss
print(f"present_loss_count : {len(pres_loss)}\n present_all_loss : {pres_loss}\n")
print(f"new_loss_count : {len(new_loss)}\n new_loss_loss : {new_loss}\n")
print(f"all_loss_count  : {all_loss} \n all_loss_count : {len(all_loss)} \n ")

all_data = pd.read_csv("./all_data_0511.csv")


# # wound - 개수 & 손실함수 누락 여부 확인

# In[149]:


len(all_data[all_data["data_name"] == "wound"]) #/28


# In[150]:


begin = all_data[all_data["data_name"] == "wound"]

M = begin["model_name"].unique().tolist()
L = begin["loss_name"].unique().tolist()

nurack_loss = []
for m in M:
    m_b = begin[begin["model_name"] == m]

    groups = m_b.groupby(["loss_name"])["iter"].agg(["count"])

    # print(f"\n{m}",groups,"\n")
    # print(f"loss count: {len(groups)}")

    loss = m_b["loss_name"].unique()

    for l in L:
        if l not in loss:
            nurack_loss.append(l)
            
    # if len(nurack_loss) != 0:
    #     print(f"\n{m}",groups,"\n")
    #     print(f"loss count: {len(groups)}")
    #     print(f"\n누락된 loss :{nurack_loss}\n")

    print(f"\n{m}",groups,"\n")
    print(f"loss count: {len(groups)}")
    print(f"\n누락된 loss :{nurack_loss}\n")
    
    nurack_loss = []


# # Kvasir - 개수 & 손실함수 누락 여부 확인

# In[151]:


len(all_data[all_data["data_name"] == "Kvasir-SEG"])


# In[152]:


begin = all_data[all_data["data_name"] == "Kvasir-SEG"]

M = begin["model_name"].unique().tolist()
L = begin["loss_name"].unique().tolist()

nurack_loss = []
for m in M:
    m_b = begin[begin["model_name"] == m]

    groups = m_b.groupby(["loss_name"])["iter"].agg(["count"])

    print(f"\n{m}",groups,"\n")
    print(f"loss count: {len(groups)}")

    loss = m_b["loss_name"].unique()

    for l in L:
        if l not in loss:
            nurack_loss.append(l)
    print(f"\n{nurack_loss}\n")
    
    nurack_loss = []


# In[153]:


begin = all_data[all_data["data_name"] == "Kvasir-SEG"]

M = begin["model_name"].unique().tolist()
L = begin["loss_name"].unique().tolist()

nurack_loss = []
for m in M:
    m_b = begin[begin["model_name"] == m]

    groups = m_b.groupby(["loss_name"])["iter"].agg(["count"])

    # print(f"\n{m}",groups,"\n")
    # print(f"loss count: {len(groups)}")

    loss = m_b["loss_name"].unique()

    for l in L:
        if l not in loss:
            nurack_loss.append(l)
            
    if len(nurack_loss) != 0:
        print(f"\n{m}",groups,"\n")
        print(f"loss count: {len(groups)}")
        print(f"\n누락된 loss :{nurack_loss}\n")
    
    nurack_loss = []


# # begin - 개수 & 손실함수 누락 여부 확인

# In[154]:


begin_len = len(all_data[all_data["data_name"] == "breast-cancer-benign"])
print(f"총 개수 : {begin_len}")


# In[155]:


begin = all_data[all_data["data_name"] == "breast-cancer-benign"]

M = begin["model_name"].unique().tolist()
L = begin["loss_name"].unique().tolist()

nurack_loss = []
for m in M:
    m_b = begin[begin["model_name"] == m]

    groups = m_b.groupby(["loss_name"])["iter"].agg(["count"])

    # print(f"\n{m}",groups,"\n")
    # print(f"loss count: {len(groups)}")

    loss = m_b["loss_name"].unique()

    for l in L:
        if l not in loss:
            nurack_loss.append(l)
            
    # if len(nurack_loss) != 0:
    print(f"\n{m}",groups,"\n")
    print(f"loss count: {len(groups)}")
    print(f"\n누락된 loss :{nurack_loss}\n")
    
    nurack_loss = []


# # CVC - 개수 & 손실함수 누락 여부 확인

# In[156]:


cvc_len = len(all_data[all_data["data_name"] == "CVC-ClinicDB"])
cvc_len


# In[157]:


begin = all_data[all_data["data_name"] == "CVC-ClinicDB"]
M = begin["model_name"].unique().tolist()
L = begin["loss_name"].unique().tolist()

nurack_loss = []
for m in M:
    m_b = begin[begin["model_name"] == m]

    groups = m_b.groupby(["loss_name"])["iter"].agg(["count"])

    loss = m_b["loss_name"].unique()

    for l in L:
        if l not in loss:
            nurack_loss.append(l)
            
    print(f"\n{m}",groups,"\n")
    print(f"loss count: {len(groups)}")
    print(f"\n누락된 loss :{nurack_loss}\n")
        
    # if len(nurack_loss) != 0:
    #     print(f"\n{m}",groups,"\n")
    #     print(f"loss count: {len(groups)}")
    #     print(f"\n누락된 loss :{nurack_loss}\n")
    
    nurack_loss = []


# # malignant - 개수 & 손실함수 누락 여부 확인

# In[158]:


malignant_len = len(all_data[all_data["data_name"] == "breast-cancer-malignant"])
malignant_len


# In[159]:


malignant = all_data[all_data["data_name"] == "breast-cancer-malignant"]

M = malignant["model_name"].unique().tolist()
L = malignant["loss_name"].unique().tolist()

nurack_loss = []
for m in M:
    m_b = malignant[malignant["model_name"] == m]

    groups = m_b.groupby(["loss_name"])["iter"].agg(["count"])

    loss = m_b["loss_name"].unique()

    for l in L:
        if l not in loss:
            nurack_loss.append(l)

    # if len(nurack_loss) != 0:
    print(f"\n{m}",groups,"\n")
    print(f"loss count: {len(groups)}")
    print(f"\n누락된 loss :{nurack_loss}\n")
    
    nurack_loss = []


# # ISIC - 완료

# In[160]:


ISIC_len = len(all_data[all_data["data_name"] == "ISIC"])
ISIC_len


# In[161]:


ISIC = all_data[all_data["data_name"] == "ISIC"]

M = ISIC["model_name"].unique().tolist()
L = ISIC["loss_name"].unique().tolist()

nurack_loss = []
for m in M:
    m_b = ISIC[ISIC["model_name"] == m]

    groups = m_b.groupby(["loss_name"])["iter"].agg(["count"])

    loss = m_b["loss_name"].unique()

    for l in L:
        if l not in loss:
            nurack_loss.append(l)

    
    print(f"\n{m}",groups,"\n")
    print(f"loss count: {len(groups)}")
    print(f"\n누락된 loss :{nurack_loss}\n")
    
    nurack_loss = []


# # 모델 - loss

# In[187]:


all_data


# In[204]:


all_data[~all_data["loss_name"].str.contains("Weighted", na=False)]


# In[200]:


all_data["loss_name"].unique()


# In[205]:


df = all_data.copy()

df_no_weight = df[~df["loss_name"].str.contains("Weighted", na=False)]

# Pivot the table where 'data_name' is the index, 'loss_name' are columns, and values are the mean of other columns
pivot_table_1 = df_no_weight.pivot_table(index='data_name', columns='loss_name', values='test_dice', aggfunc='mean')

pivot_table_1


# In[208]:


total_rank_df = pd.DataFrame(data = [0 for _ in range(len(pivot_table_1.columns))],
                             index=[loss for loss in list(pivot_table_1.columns)]).T

data_piv_1= pivot_table_1

all_loss_numb = len(pivot_table_1.columns) # 34
ranking = 10

for data in data_piv_1.index:
    for i in range(all_loss_numb):
        sort = data_piv_1.loc[data].sort_values(ascending=False)[:]
        ranks = ranking - i
        idx = sort.index[i]
        
        if ranks >=0:
            total_rank_df[idx]+=ranks
        else:
            total_rank_df[idx] = 0
        
        # total_rank_df[idx]+=ranks
        # if i+1 == 1:
        #     total_rank_df[idx]+=ranks
        # elif i+1 == 2:    
        #     total_rank_df[idx]+=ranks
        # elif i+1 == 3:
        #     total_rank_df[idx]+=ranks
        # elif i+1 == 4:    
        #     total_rank_df[idx]+=ranks
        # elif i+1 == 5:
        #     total_rank_df[idx]+=ranks
            
data_piv_1.loc['total'] = list(total_rank_df.loc[0].values)

pivot_table_sort_1 = data_piv_1.copy()
pivot_table_sort_1 = pivot_table_sort_1.T.sort_values(by = 'total',ascending=False).T
pivot_table_sort_1 = pivot_table_sort_1.fillna(0.0)
pivot_table_sort_1


# In[209]:


# pivot_table_sort_1.columns = pivot_table_sort_1.columns[1:]
testing = pivot_table_sort_1.T.sort_values(by = 'total',ascending=False).copy()
plt.rcdefaults()

nrank = 10 # You can change this value to display up to 'nrank' ranks

# Assume pivot_table is already defined
data = testing.values
data_ind = testing.columns           

# Calculate ranks per row
ranks = np.zeros_like(data, dtype=int)
sorted_indices = np.argsort(-data, axis=0)

for i in range(data.shape[1]):
    ranks[sorted_indices[:,i],i] = np.arange(1, data.shape[0] + 1)

# Transform rank data for color mapping
ranked_data = np.where(ranks <= nrank, ranks, nrank + 1)

# Use ColorBrewer's 'Reds' colormap
cmap = plt.get_cmap('Reds')

# Generate colors for 'nrank' ranks
color_positions = np.linspace(0.8, 0.2, nrank)
colors = [cmap(pos) for pos in color_positions]
colors.append('#F9F9F9')  # Color for 'Others'

# Create the new colormap
new_cmap = mcolors.ListedColormap(colors)

# Set normalization
bounds = np.arange(1, nrank + 3)
norm = mcolors.BoundaryNorm(bounds, new_cmap.N)

# Plot the heatmap
plt.figure(figsize=(7, 7))
ax = sns.heatmap(
    ranked_data,
    annot=data,
    fmt=".4f",
    cmap=new_cmap,
    norm=norm,
    cbar=False,
    xticklabels=testing.columns,
    yticklabels=testing.index,
    linewidths=0.5,
    annot_kws={"size" :9},
)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)  # x축 글자 크기
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, rotation=0)  # y축 글자 크기


plt.tight_layout()       
out_path = f"./heat_map_barplot/result_data_heat.svg"
plt.savefig(out_path, format="svg", bbox_inches="tight")

plt.show()


# # 

# In[212]:


total_rank_df = pd.DataFrame(data = [0 for _ in range(len(pivot_table_1.columns))],
                             index=[loss for loss in list(pivot_table_1.columns)]).T

data_piv_1= pivot_table_1

all_loss_numb = len(pivot_table_1.columns) # 34
ranking = 10


# In[216]:


# Pivot the table where 'data_name' is the index, 'loss_name' are columns, and values are the mean of other columns
pivot_table_2 = df_no_weight.pivot_table(index='model_name', columns='loss_name', values='test_dice', aggfunc='mean')
pivot_table_2 = pivot_table_2.fillna(0.0)
pivot_table_2

total_rank_df = pd.DataFrame(data = [0 for _ in range(len(pivot_table_1.columns))],
                             index=[loss for loss in list(pivot_table_1.columns)]).T
data_piv_2= pivot_table_2
all_loss_numb = len(pivot_table_1.columns) # 34

ranking = 10
for model in data_piv_2.index:
    for i in range(all_loss_numb):
        sort = data_piv_2.loc[model].sort_values(ascending=False)[:]
        idx = sort.index[i]

        ranks = ranking - i
        if ranks>=0:
            total_rank_df[idx]+=ranks
        
        else:
            ranks = 0
            total_rank_df[idx]+=ranks
            
data_piv_2.loc['total'] = list(total_rank_df.loc[0].values)

pivot_table_sort_2 = data_piv_2.copy()
pivot_table_sort_2 = pivot_table_sort_2.T.sort_values(by = 'total',ascending=False).T
pivot_table_sort_2 = pivot_table_sort_2.fillna(0.0)
pivot_table_sort_2


# In[218]:


testing = pivot_table_sort_2.T.sort_values(by = 'total',ascending=False).copy()
plt.rcdefaults()

nrank = 10 # You can change this value to display up to 'nrank' ranks

# Assume pivot_table is already defined
data = testing.values
data_ind = testing.columns           

# Calculate ranks per row
ranks = np.zeros_like(data, dtype=int)
sorted_indices = np.argsort(-data, axis=0)

for i in range(data.shape[1]):
    ranks[sorted_indices[:,i],i] = np.arange(1, data.shape[0] + 1)

# Transform rank data for color mapping
ranked_data = np.where(ranks <= nrank, ranks, nrank + 1)

# Use ColorBrewer's 'Reds' colormap
cmap = plt.get_cmap('Reds')

# Generate colors for 'nrank' ranks
color_positions = np.linspace(0.8, 0.2, nrank)
colors = [cmap(pos) for pos in color_positions]
colors.append('#F9F9F9')  # Color for 'Others'

# Create the new colormap
new_cmap = mcolors.ListedColormap(colors)

# Set normalization
bounds = np.arange(1, nrank + 3)
norm = mcolors.BoundaryNorm(bounds, new_cmap.N)

# Plot the heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    ranked_data,
    annot=data,
    fmt=".4f",
    cmap=new_cmap,
    norm=norm,
    cbar=False,
    xticklabels=testing.columns,
    yticklabels=testing.index,
    linewidths=0.5,
    annot_kws={"size" :9},
)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)  # x축 글자 크기
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, rotation=0)  # y축 글자 크기


plt.show()


# # Barplot

# ## 결합(주변지역정보랑 구분X) VS. 단일

# In[231]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

metrics = ['test_dice', 'test_iou', 'test_precision', 'test_recall']
metrics_len = len(metrics)

W = 61
H = 4

df_new = df_no_weight[df_no_weight['model_name'] != 'SegResNet']
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(25, 40))

for i1 in range(H):   
    # fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(25, 10))
    losses = df_new['loss_name'].unique()
    mean_values = []

    for loss in losses:   
        vals = df_new[df_new['loss_name'] == loss][metrics[i1]].dropna()
        mean_val = vals.mean()
        mean_values.append(mean_val)

    # 데이터프레임 생성
    data = pd.DataFrame(data=mean_values, index=list(losses), columns=['value_mean'])
    
    # NaN 제거 (혹시 평균이 NaN인 경우)
    data = data.dropna(subset=['value_mean'])

    colors_1 = sns.color_palette("ch:s=.25,rot=-.25")[0]
    colors_2 = sns.color_palette("ch:s=.25,rot=-.25")[1]

    data['color'] = [colors_1 if loss in ['BCELoss', 'DiceLoss', 'FocalLoss', 'IoULoss'] else colors_2 for loss in data.index]
    data = data.sort_values(by='value_mean', ascending=True)

    # 수평 바 차트 그리기
    axs[i1].barh(data.index, data['value_mean'], color=data['color'])

    # 성능 지표 이름 설정
    m = [metric.replace('test_', '') for metric in metrics]
    reset_met = [m[i][0].upper() + m[i][1:] if m[i] != 'iou' else 'IoU' for i in range(metrics_len)]
    axs[i1].set_title(f"{reset_met[i1]}", fontsize=15)

    # 바 위에 수치 표시
    for j, (value, y) in enumerate(zip(data['value_mean'], data.index)):
        min_loc = data['value_mean'].min()
        max_loc = data['value_mean'].max()
        vas = value - 0.0000001

        axs[i1].tick_params(axis='x', labelsize=13)
        axs[i1].tick_params(axis='y', labelsize=13)
        axs[i1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
        axs[i1].text(vas, j, f'{value:.4f}', va='center', fontsize=11.5)
        axs[i1].set_xlim(min_loc - 0.01, max_loc + 0.02)

    plt.tight_layout()
    plt.savefig(f"./heat_map_barplot/bar_orignal_{reset_met[i1]}.svg", format="svg", bbox_inches="tight")
plt.savefig(f"./heat_map_barplot/ALL_bar_orignal.svg", format="svg", bbox_inches="tight")


# ## 결합(주변지역정보 O) VS. 결합(주변지역정보 X) VS. 단일

# In[234]:


metrics = ['test_dice', 'test_iou', 'test_precision', 'test_recall']
metrics_len = len(metrics)

W = 61
H = 4

df_new = df_no_weight[df_no_weight['model_name'] != 'SegResNet']
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(25, 40))
for i1 in range(H):   
    # fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(25, 10))
    losses = df_new['loss_name'].unique()
    mean_values = []

    for loss in losses:   
        vals = df_new[df_new['loss_name'] == loss][metrics[i1]].dropna()
        mean_val = vals.mean()
        mean_values.append(mean_val)

    # 데이터프레임 생성
    data = pd.DataFrame(data=mean_values, index=list(losses), columns=['value_mean'])
    
    # NaN 제거 (혹시 평균이 NaN인 경우)
    data = data.dropna(subset=['value_mean'])

    colors_1 = sns.color_palette("ch:s=.25,rot=-.25")[0]
    colors_2 = sns.color_palette("ch:s=.25,rot=-.25")[1]
    colors_3 = sns.color_palette("ch:s=.25,rot=-.25")[2]

    colors = []
    for loss in data.index:
        if any(k in loss for k in ['GaussianLoss', 'TVLoss', 'EdgeLoss', 'ContextualLoss', 'WeightedLoss']):
            colors.append(colors_3)
            
        elif any(k == loss for k in ['BCELoss', 'DiceLoss', 'FocalLoss', 'IoULoss']):
            colors.append(colors_1)
        else:
            colors.append(colors_2)
    data['color'] = colors
    data = data.sort_values(by='value_mean', ascending=True)

    # 수평 바 차트 그리기
    axs[i1].barh(data.index, data['value_mean'], color=data['color'])

    # 성능 지표 이름 설정
    m = [metric.replace('test_', '') for metric in metrics]
    reset_met = [m[i][0].upper() + m[i][1:] if m[i] != 'iou' else 'IoU' for i in range(metrics_len)]
    axs[i1].set_title(f"{reset_met[i1]}", fontsize=15)

    # 바 위에 수치 표시
    for j, (value, y) in enumerate(zip(data['value_mean'], data.index)):
        min_loc = data['value_mean'].min()
        max_loc = data['value_mean'].max()
        vas = value - 0.0000001

        axs[i1].tick_params(axis='x', labelsize=13)
        axs[i1].tick_params(axis='y', labelsize=13)
        axs[i1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
        axs[i1].text(vas, j, f'{value:.4f}', va='center', fontsize=11.5)
        axs[i1].set_xlim(min_loc - 0.01, max_loc + 0.02)

    plt.tight_layout()
    plt.savefig(f"./heat_map_barplot/bar_gubun_{reset_met[i1]}.svg", format="svg", bbox_inches="tight")
plt.savefig(f"./heat_map_barplot/ALL_bar_gubun.svg", format="svg", bbox_inches="tight")

