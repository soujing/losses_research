import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel, device):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1).to(device)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).to(device))
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    padding = window_size // 2
    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

    # `torch.FloatTensor`로 변환 불필요. 이미 텐서인 경우 GPU에 있는지 확인만 하면 됨.
    mu1 = mu1.to(img1.device)
    mu2 = mu2.to(img2.device)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = None  # 초기화에서 생성하지 않음

    def forward(self, img1, img2):
        device = img1.device  # 입력 텐서의 디바이스 확인
        (_, channel, _, _) = img1.size()

        # `self.window` 생성 시 디바이스 확인
        if self.window is None or channel != self.channel or self.window.device != device:
            self.window = create_window(self.window_size, channel, device)
            self.channel = channel

        return _ssim(img1, img2, self.window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size=11, size_average=True):
    device = img1.device  # 입력 텐서와 동일한 디바이스 사용
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel, device)
    return _ssim(img1, img2, window, window_size, channel, size_average)
