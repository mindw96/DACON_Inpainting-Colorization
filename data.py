from skimage.metrics import structural_similarity as ski_ssim
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image

import torch.nn.functional as F
from polygenerator import (
    random_polygon,
    random_star_shaped_polygon,
    random_convex_polygon,
)
import skimage
import numpy as np
import torch
import random
import cv2
import os


def get_input_image(image, min_polygon_bbox_size=50, transform=None):
    width, height = image.size
    while True:
        bbox_x1 = random.randint(0, width - min_polygon_bbox_size)
        bbox_y1 = random.randint(0, height - min_polygon_bbox_size)
        bbox_x2 = random.randint(bbox_x1, width)  # Ensure width > 10
        bbox_y2 = random.randint(bbox_y1, height)  # Ensure height > 10
        if (bbox_x2 - bbox_x1) < min_polygon_bbox_size or (bbox_y2 - bbox_y1) < min_polygon_bbox_size:
            continue

        mask_bbox = [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
        mask_width = bbox_x2 - bbox_x1
        mask_height = bbox_y2 - bbox_y1

        num_points = random.randint(3, 20)
        polygon_func = random.choice([
            random_polygon,
            random_star_shaped_polygon,
            random_convex_polygon
        ])
        polygon = polygon_func(num_points=num_points)  # scaled 0~1
        polygon = [(round(r * mask_width), round(c * mask_height)) for r, c in polygon]
        polygon_mask = skimage.draw.polygon2mask((mask_width, mask_height), polygon)
        if np.sum(polygon_mask) > (min_polygon_bbox_size // 2) ** 2:
            break
    full_image_mask = np.zeros((width, height), dtype=np.uint8)
    full_image_mask[bbox_x1:bbox_x2, bbox_y1:bbox_y2] = polygon_mask

    image_gray = image.convert('L')
    image_gray_array = np.array(image_gray)  # Convert to numpy array for manipulation
    random_color = random.randint(0, 255)  # Random grayscale color
    image_gray_array[full_image_mask == 1] = random_color
    image_gray_masked = Image.fromarray(image_gray_array)

    return {
        'image_gt': transform(image),
        'mask': transform(full_image_mask),
        'image_gray': transform(image_gray),
        'image_gray_masked': transform(image_gray_masked)
    }


class CustomDataset(Dataset):
    def __init__(self, damage_dir, origin_dir, transform=None):
        self.damage_dir = damage_dir
        self.origin_dir = origin_dir
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.damage_files = sorted(os.listdir(damage_dir))
        self.origin_files = sorted(os.listdir(origin_dir))

    def __len__(self):
        return len(self.damage_files)

    def __getitem__(self, idx):
        damage_img_name = self.damage_files[idx]
        origin_img_name = self.origin_files[idx]

        damage_img_path = os.path.join(self.damage_dir, damage_img_name)
        origin_img_path = os.path.join(self.origin_dir, origin_img_name)

        damage_img = Image.open(damage_img_path).convert("RGB")
        origin_img = Image.open(origin_img_path).convert('RGB')

        if self.transform:
            damage_img = self.transform(damage_img)
            origin_img = self.transform(origin_img)

        return {'A': damage_img, 'B': origin_img}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_ssim_score(true, pred):
    # 전체 RGB 이미지를 사용해 SSIM 계산 (channel_axis=-1)
    ssim_value = ski_ssim(true, pred, channel_axis=-1, data_range=pred.max() - pred.min())

    return ssim_value


def ssim_score_gpu(img1, img2, window_size=11, C1=0.01 ** 2, C2=0.03 ** 2, device=None):
    """
    GPU에서 SSIM 계산
    img1, img2: [N, C, H, W] 형태의 텐서 (0~1 범위, RGB 이미지)
    window_size: Gaussian 윈도우 크기
    C1, C2: SSIM 안정성 상수
    """

    # Gaussian 윈도우 생성
    def gaussian_window(window_size, sigma=1.5, channels=1):
        """
        Gaussian 윈도우를 생성하여 반환
        window_size: 윈도우 크기
        sigma: 가우시안 분포의 표준 편차
        channels: 채널 수
        """
        gauss = torch.tensor([
            torch.exp(-torch.tensor(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ], dtype=torch.float32)  # float32로 명시적으로 텐서 생성
        gauss /= gauss.sum()  # 정규화

        kernel = gauss[:, None] @ gauss[None, :]  # 외적 계산으로 2D 가우시안 생성
        kernel /= kernel.sum()  # 정규화
        kernel = kernel.expand(channels, 1, window_size, window_size)  # 채널 확장
        return kernel

    # 윈도우 생성 및 이미지 패딩
    _, channels, height, width = img1.size()
    window = gaussian_window(window_size, channels=channels).to(device=device)
    padding = window_size // 2

    # 밝기 (Mean)
    mu1 = F.conv2d(img1, window, padding=padding, groups=channels)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channels)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 대비 (Variance)
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channels) - mu1_mu2

    # SSIM 계산
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean().item()


def get_histogram_similarity(true, pred, cvt_color=cv2.COLOR_RGB2HSV):
    # true_np = ((true * 0.5 + 0.5) * 255.0).clamp(0, 255).round().detach().cpu().permute(0, 2, 3, 1).float().numpy().astype(np.uint8)
    # true_np = ((true * 0.5 + 0.5) * 255.0).clamp(0, 255).round().detach().cpu().float().numpy().astype(np.uint8)
    # pred_np = ((pred * 0.5 + 0.5) * 255.0).clamp(0, 255).round().detach().cpu().permute(0, 2, 3, 1).float().numpy().astype(np.uint8)

    # BGR 이미지를 HSV로 변환
    true_hsv = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2HSV) for img in true])
    # true_hsv = cv2.cvtColor(true, cv2.COLOR_RGB2HSV)
    pred_hsv = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2HSV) for img in pred])
    # pred_hsv = cv2.cvtColor(pred, cv2.COLOR_RGB2HSV)

    # H 채널에서 히스토그램 계산 및 정규화
    hist_true = cv2.calcHist([true_hsv], [0], None, [180], [0, 180])
    hist_pred = cv2.calcHist([pred_hsv], [0], None, [180], [0, 180])
    hist_true = cv2.normalize(hist_true, hist_true).flatten()
    hist_pred = cv2.normalize(hist_pred, hist_pred).flatten()

    # 히스토그램 간 유사도 계산 (상관 계수 사용)
    similarity = cv2.compareHist(hist_true, hist_pred, cv2.HISTCMP_CORREL)

    return similarity


def histogram_similarity_gpu(true, pred, num_bins=180):
    """
    GPU에서 HSV 색상 채널의 히스토그램 유사도를 계산
    true, pred: [N, C, H, W] 형태의 텐서 (RGB 이미지, 값 범위 0~1)
    num_bins: 히스토그램의 bin 수 (기본값: 180)
    """

    # RGB 이미지를 HSV로 변환
    def rgb_to_hsv(img):
        """
        RGB 텐서를 HSV 텐서로 변환
        img: [N, C, H, W] 형태의 텐서
        반환: [N, C, H, W] 형태의 HSV 텐서
        """
        r, g, b = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

        max_val, _ = img.max(dim=1, keepdim=True)  # [N, 1, H, W]
        min_val, _ = img.min(dim=1, keepdim=True)  # [N, 1, H, W]
        delta = max_val - min_val

        # Hue 계산
        hue = torch.zeros_like(max_val)  # [N, 1, H, W]
        mask = delta != 0  # [N, 1, H, W]

        max_val_r = (max_val == r.unsqueeze(1))  # [N, 1, H, W]
        max_val_g = (max_val == g.unsqueeze(1))  # [N, 1, H, W]
        max_val_b = (max_val == b.unsqueeze(1))  # [N, 1, H, W]

        # Red 최대
        hue[mask & max_val_r] = ((60 * (g - b)).unsqueeze(1) / delta)[mask & max_val_r] % 360
        # Green 최대
        hue[mask & max_val_g] = ((60 * (b - r)).unsqueeze(1) / delta + 120)[mask & max_val_g]
        # Blue 최대
        hue[mask & max_val_b] = ((60 * (r - g)).unsqueeze(1) / delta + 240)[mask & max_val_b]

        hue = hue / 360  # 0~1 범위로 정규화

        # Saturation 계산
        saturation = torch.zeros_like(max_val)
        saturation[max_val != 0] = delta[max_val != 0] / max_val[max_val != 0]

        # Value 계산
        value = max_val

        return torch.cat((hue, saturation, value), dim=1)  # [N, 3, H, W]

    # HSV 변환
    true_hsv = rgb_to_hsv(true)
    pred_hsv = rgb_to_hsv(pred)

    # H 채널 히스토그램 계산
    def calculate_histogram(h_channel, num_bins):
        """
        H 채널에서 히스토그램 계산
        h_channel: [N, H, W] 형태의 텐서
        num_bins: 히스토그램의 bin 수
        반환: [N, num_bins] 형태의 히스토그램 텐서
        """
        batch_size = h_channel.size(0)
        h_channel = (h_channel * num_bins).long()  # 0~1 범위를 0~num_bins로 변환
        hist = torch.zeros(batch_size, num_bins, device=h_channel.device)

        for i in range(batch_size):
            hist[i] = torch.histc(h_channel[i].float(), bins=num_bins, min=0, max=num_bins - 1)

        hist = hist / hist.sum(dim=1, keepdim=True)  # 정규화
        return hist

    hist_true = calculate_histogram(true_hsv[:, 0, :, :], num_bins)
    hist_pred = calculate_histogram(pred_hsv[:, 0, :, :], num_bins)

    # 히스토그램 유사도 계산 (코사인 유사도 사용)
    similarity = torch.sum(hist_true * hist_pred, dim=1)  # [N]

    return similarity.mean().item()


def get_masked_ssim_score(true, pred, mask):
    # 손실 영역의 좌표에서만 RGB 채널별 픽셀 값 추출
    true_masked_pixels = true[mask > 0]
    pred_masked_pixels = pred[mask > 0]

    # 손실 영역 픽셀만으로 SSIM 계산 (채널축 사용)
    ssim_value = ski_ssim(
        true_masked_pixels,
        pred_masked_pixels,
        channel_axis=-1,
        data_range=pred.max() - pred.min()
    )
    return ssim_value
