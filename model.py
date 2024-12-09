import cv2
import numpy as np

from data import get_ssim_score, get_histogram_similarity, ssim_score_gpu, histogram_similarity_gpu, \
    get_masked_ssim_score
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.models import vgg16
from torch import nn

import segmentation_models_pytorch as smp
from torch.autograd import grad
import torch.nn.functional as F
import lightning as L
import pytorch_ssim
import torchvision
import torch


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        attention = F.softmax(torch.bmm(proj_query, proj_key), dim=-1)
        proj_value = self.value(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.global_avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.global_max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat_out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat_out)
        return self.sigmoid(out) * x


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg[:16])).eval()  # Conv4_3 layer
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input = input.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
        input_features = self.feature_extractor(input)
        target_features = self.feature_extractor(target)
        return nn.functional.l1_loss(input_features, target_features)


class InpaintingGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(InpaintingGenerator, self).__init__()

        def down_block(in_feat, out_feat, normalize=True, use_spectral=False):
            layers = []
            conv_layer = nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1)
            if use_spectral:
                conv_layer = nn.utils.spectral_norm(conv_layer)
            layers.append(conv_layer)
            if normalize:
                layers.append(nn.InstanceNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up_block(in_feat, out_feat, dropout=0.0, use_attention=False, use_cbam=False):
            layers = [
                nn.ConvTranspose2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_feat),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            if dropout:
                layers.append(nn.Dropout(dropout))
            if use_attention:
                layers.append(SelfAttention(out_feat))  # 기존 Self-Attention 사용
            if use_cbam:
                layers.append(CBAM(out_feat))  # CBAM 추가
            return nn.Sequential(*layers)

        self.down1 = down_block(in_channels, 64, normalize=False, use_spectral=True)
        self.down2 = down_block(64, 128, use_spectral=True)
        self.down3 = down_block(128, 256, use_spectral=True)
        self.down4 = down_block(256, 512, use_spectral=True)
        self.down5 = down_block(512, 512, use_spectral=True)
        self.down6 = down_block(512, 512, use_spectral=True)
        self.down7 = down_block(512, 512, use_spectral=True)
        self.down8 = down_block(512, 512, normalize=False, use_spectral=True)

        self.up1 = up_block(512, 512, dropout=0.5, use_cbam=True)
        self.up2 = up_block(1024, 512, dropout=0.5, use_cbam=True)
        self.up3 = up_block(1024, 512, dropout=0.5, use_cbam=True)
        self.up4 = up_block(1024, 512, use_cbam=True)
        self.up5 = up_block(1024, 256, use_cbam=True)
        self.up6 = up_block(512, 128, use_cbam=True)
        self.up7 = up_block(256, 64, use_cbam=True)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        u8 = self.up8(torch.cat([u7, d1], 1))

        return u8


class ColorizationGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(ColorizationGenerator, self).__init__()

        def down_block(in_feat, out_feat, normalize=True, use_spectral=False):
            layers = []
            conv_layer = nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1)
            if use_spectral:
                conv_layer = nn.utils.spectral_norm(conv_layer)
            layers.append(conv_layer)
            if normalize:
                layers.append(nn.InstanceNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up_block(in_feat, out_feat, dropout=0.0, use_attention=False, use_cbam=False):
            layers = [
                nn.ConvTranspose2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_feat),
                nn.ReLU(inplace=True)
            ]
            if dropout:
                layers.append(nn.Dropout(dropout))
            if use_attention:
                layers.append(SelfAttention(out_feat))
            if use_cbam:
                layers.append(CBAM(out_feat))  # CBAM 모듈 추가
            return nn.Sequential(*layers)

        self.down1 = down_block(in_channels, 64, normalize=False, use_spectral=True)
        self.down2 = down_block(64, 128, use_spectral=True)
        self.down3 = down_block(128, 256, use_spectral=True)
        self.down4 = down_block(256, 512, use_spectral=True)
        self.down5 = down_block(512, 512, use_spectral=True)
        self.down6 = down_block(512, 512, use_spectral=True)
        self.down7 = down_block(512, 512, use_spectral=True)
        self.down8 = down_block(512, 512, normalize=False, use_spectral=True)

        self.up1 = up_block(512, 512, dropout=0.5, use_cbam=True)
        self.up2 = up_block(1024, 512, dropout=0.5, use_cbam=True)
        self.up3 = up_block(1024, 512, dropout=0.5, use_cbam=True)
        self.up4 = up_block(1024, 512, use_cbam=True)
        self.up5 = up_block(1024, 256, use_cbam=True)
        self.up6 = up_block(512, 128, use_cbam=True)
        self.up7 = up_block(256, 64, use_cbam=True)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        u8 = self.up8(torch.cat([u7, d1], 1))

        return u8


class InpaintingDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(InpaintingDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, use_spectral=False, normalization=True):
            layers = []
            conv_layer = nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)
            if use_spectral:
                conv_layer = nn.utils.spectral_norm(conv_layer)
            layers.append(conv_layer)
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # Downsampling layers
        self.block1 = discriminator_block(in_channels * 2, 64, normalization=False, use_spectral=True)
        self.block2 = discriminator_block(64, 128, use_spectral=True)
        self.block3 = discriminator_block(128, 256, use_spectral=True)
        self.block4 = discriminator_block(256, 512, use_spectral=True)

        # Final convolution layer
        self.final_layer = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

        # Self-Attention layer
        self.attention = SelfAttention(256)

    def forward(self, img_A, img_B):
        # Concatenate input images
        img_input = torch.cat((img_A, img_B), 1)

        # Downsampling
        x = self.block1(img_input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.attention(x)  # Apply Self-Attention
        x = self.block4(x)

        # Output single-channel prediction map
        validity = self.final_layer(x)
        return validity


class ColorizationDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(ColorizationDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, use_spectral=False, normalization=True):
            layers = []
            conv_layer = nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)
            if use_spectral:
                conv_layer = nn.utils.spectral_norm(conv_layer)
            layers.append(conv_layer)
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # Downsampling layers
        self.block1 = discriminator_block(4, 64, normalization=False, use_spectral=True)
        self.block2 = discriminator_block(64, 128, use_spectral=True)
        self.block3 = discriminator_block(128, 256, use_spectral=True)
        self.block4 = discriminator_block(256, 512, use_spectral=True)

        # Final convolution layer
        self.final_layer = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

        # Self-Attention layer
        self.attention = SelfAttention(256)

    def forward(self, img_A, img_B):
        # Concatenate input images
        img_input = torch.cat((img_A, img_B), 1)

        # Downsampling
        x = self.block1(img_input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.attention(x)  # Apply Self-Attention
        x = self.block4(x)

        # Output single-channel prediction map
        validity = self.final_layer(x)
        return validity


class InpaintingGAN(L.LightningModule):
    def __init__(
            self,
            in_channels=1,
            out_channels=1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = InpaintingGenerator(in_channels, out_channels)
        self.discriminator = InpaintingDiscriminator(in_channels)
        self.criterionGAN = nn.BCEWithLogitsLoss()
        self.pixelwiseLoss = nn.L1Loss()
        self.ssim_Loss = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.perceptual_Loss = PerceptualLoss()

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch):
        image_gt, mask, image_gray, image_gray_masked = batch['image_gt'], batch['mask'], batch['image_gray'], batch[
            'image_gray_masked']
        optimizer_g, optimizer_d = self.optimizers()

        # generate images
        fake_B = self.generator(image_gray_masked)

        # log sampled images
        grid = torchvision.utils.make_grid((fake_B[:6] * 0.5 + 0.5).float())
        self.logger.experiment.add_image("train/inpainting_images", grid, self.current_epoch)

        # train generator
        pred_fake = self.discriminator(fake_B, image_gray)

        self.toggle_optimizer(optimizer_g)

        loss_GAN = self.criterionGAN(pred_fake, torch.ones_like(pred_fake))
        loss_pixel = self.pixelwiseLoss(fake_B, image_gray)

        self.log("gan loss", loss_GAN, prog_bar=True)
        self.log("pixel loss", loss_pixel, prog_bar=True)
        loss_G = 10 * loss_GAN + 10 * loss_pixel

        self.log("g_loss", loss_G, prog_bar=True)
        self.manual_backward(loss_G)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        self.toggle_optimizer(optimizer_d)

        pred_real = self.discriminator(image_gray, image_gray)
        loss_real = self.criterionGAN(pred_real, torch.rand_like(pred_real) * 0.2 + 0.8)

        pred_fake = self.discriminator(fake_B.detach(), image_gray)
        loss_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))

        loss_D = 0.5 * (loss_real + loss_fake)
        self.log("d_loss", loss_D, prog_bar=True)
        self.manual_backward(loss_D)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

        scheduler_G = torch.optim.lr_scheduler.StepLR(opt_g, step_size=5, gamma=0.5)
        scheduler_D = torch.optim.lr_scheduler.StepLR(opt_d, step_size=5, gamma=0.5)

        return [opt_g, opt_d], [scheduler_G, scheduler_D]


class ColorizationGAN(L.LightningModule):
    def __init__(
            self,
            in_channels=1,
            out_channels=3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = ColorizationGenerator(in_channels, out_channels)
        self.discriminator = ColorizationDiscriminator(in_channels)
        self.criterionGAN = nn.BCEWithLogitsLoss()
        self.pixelwiseLoss = nn.L1Loss()

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch):
        image_gt, mask, image_gray, image_gray_masked = batch['image_gt'], batch['mask'], batch['image_gray'], batch[
            'image_gray_masked']
        optimizer_g, optimizer_d = self.optimizers()
        # sch_g, sch_d = self.lr_schedulers()

        # generate images
        self.toggle_optimizer(optimizer_g)
        fake_B = self.generator(image_gray)

        # log sampled images
        sample_imgs = (fake_B[:6] * 0.5 + 0.5).float()
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("train/generated_images", grid, self.current_epoch)

        pred_fake = self.discriminator(fake_B, image_gray)
        loss_GAN = self.criterionGAN(pred_fake, torch.ones_like(pred_fake))
        loss_pixel = self.pixelwiseLoss(fake_B, image_gt)
        self.log("gan loss", loss_GAN, prog_bar=True)
        self.log("pixel loss", loss_pixel, prog_bar=True)
        loss_G = loss_GAN + 10 * loss_pixel

        self.log("g_loss", loss_G, prog_bar=True)
        self.manual_backward(loss_G)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        # sch_g.step()

        # train discriminator
        self.toggle_optimizer(optimizer_d)

        pred_real = self.discriminator(image_gt, image_gray)
        loss_real = self.criterionGAN(pred_real, torch.ones_like(pred_real) * 0.9)

        pred_fake = self.discriminator(fake_B.detach(), image_gray)
        loss_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))

        loss_D = 0.5 * (loss_real + loss_fake)
        self.log("d_loss", loss_D, prog_bar=True)
        self.manual_backward(loss_D)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

        scheduler_G = torch.optim.lr_scheduler.StepLR(opt_g, step_size=5, gamma=0.5)
        scheduler_D = torch.optim.lr_scheduler.StepLR(opt_d, step_size=5, gamma=0.5)

        return [opt_g, opt_d], [scheduler_G, scheduler_D]
