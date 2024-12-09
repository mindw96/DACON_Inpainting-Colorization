
from model import InpaintingGAN, ColorizationGAN
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm

import segmentation_models_pytorch as smp
import numpy as np
import zipfile
import torch
import cv2
import os


def load_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize([0.5], [0.5])
    ])

    image = Image.open(image_path).convert('L')
    image = transform(image)
    image = image.unsqueeze(0)  # 배치 차원을 추가합니다.

    return image


def main():
    test_dir = './test_clean_gray'

    inpainting_checkpoint = "lightning_logs/version_123/checkpoints/epoch=19-step=14840.ckpt"
    inpainting_model = InpaintingGAN.load_from_checkpoint(inpainting_checkpoint)
    inpainting_model.eval()
    #
    colorization_checkpoint = "lightning_logs/version_28/checkpoints/epoch=49-step=30850.ckpt"
    colorization_model = ColorizationGAN.load_from_checkpoint(colorization_checkpoint)
    colorization_model.eval()

    # 저장할 디렉토리 설정
    submission_dir = "./submission5"
    os.makedirs(submission_dir, exist_ok=True)

    # 파일 리스트 불러오기
    test_images = sorted(os.listdir(test_dir))

    # 모든 테스트 이미지에 대해 추론 수행
    for image_name in tqdm(test_images):
        test_image_path = os.path.join(test_dir, image_name)

        # 손상된 테스트 이미지 로드 및 전처리
        test_image = load_image(test_image_path)

        with torch.no_grad():
            # 모델로
            test_image = test_image.to('cuda')
            # inpainting_output = inpainting_model(test_image)
            # output = colorization_model(inpainting_output)
            output = colorization_model(test_image)
            output = output.cpu().squeeze(0)
            output = output * 0.5 + 0.5
            output = output.numpy().transpose((1, 2, 0))
            output = (output * 255).astype(np.uint8)

            # 예측된 이미지를 실제 이미지와 같은 512x512로 리사이즈
            output_resized = cv2.resize(output, (512, 512), interpolation=cv2.INTER_LINEAR)

        # 결과 이미지 저장
        output_path = os.path.join(submission_dir, image_name)
        cv2.imwrite(output_path, cv2.cvtColor(output_resized, cv2.COLOR_RGB2BGR))

    print(f"Saved all images")

    # 저장된 결과 이미지를 ZIP 파일로 압축
    zip_filename = "submission.zip"
    with zipfile.ZipFile(zip_filename, 'w') as submission_zip:
        for image_name in test_images:
            image_path = os.path.join(submission_dir, image_name)
            submission_zip.write(image_path, arcname=image_name)

    print(f"All images saved in {zip_filename}")


if __name__ == '__main__':
    main()