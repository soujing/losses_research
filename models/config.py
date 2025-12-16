import torch

# 사용할 디바이스 지정 (cuda:2 또는 cpu)
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
