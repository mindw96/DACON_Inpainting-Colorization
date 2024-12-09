from model import ColorizationGAN, InpaintingGAN
from data import seed_everything, CustomDataset
from torch.utils.data import DataLoader

import lightning as L
import torch


def main():
    torch.set_float32_matmul_precision("high")

    CFG = {
        'EPOCHS': 50,
        'BATCH_SIZE': 48,
        'SEED': 9608
    }

    seed_everything(CFG['SEED'])

    origin_dir = './train_gt'  # 원본 이미지 폴더 경로
    damage_dir = './train_input'  # 손상된 이미지 폴더 경로

    # 데이터셋 및 DataLoader 생성
    dataset = CustomDataset(damage_dir=damage_dir, origin_dir=origin_dir)
    dataloader = DataLoader(dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=11,
                            persistent_workers=True)

    inpainting_model = InpaintingGAN()
    inpainting_trainer = L.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=CFG['EPOCHS'],
        precision="bf16-mixed"
    )
    inpainting_trainer.fit(inpainting_model, dataloader)

    colorization_model = ColorizationGAN()
    colorization_trainer = L.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=CFG['EPOCHS'],
        precision="bf16-mixed",
    )
    colorization_trainer.fit(colorization_model, dataloader)


if __name__ == '__main__':
    main()
