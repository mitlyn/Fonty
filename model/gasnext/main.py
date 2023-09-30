from random import randint
from typing import Optional
from lightning import LightningModule
from torchmetrics import MetricCollection
import torch, torch.nn as nn, torch.optim as op

from model.ftransgan import Generator
from model.share import Discriminator, Hinge, WGANGP
from model.types import TrainBundle, Options
from model.utils import setInit, setGrads

# *----------------------------------------------------------------------------*

class GasNeXt(LightningModule):
    def __init__(self, opt: Options, metrics: Optional[MetricCollection] = None):
        super(GasNeXt, self).__init__()

        self.automatic_optimization = False

        self.num_block = opt.num_block
        self.block_size = opt.block_size

        self.G = Generator(opt.G_filters, blocks=6, dropout=opt.G_dropout)
        setInit(self.G, opt.init_type, opt.init_gain)

        self.Dc = Discriminator(2, opt.D_filters, opt.D_layers)
        setInit(self.Dc, opt.init_type, opt.init_gain)

        self.Ds = Discriminator(opt.refs + 1, opt.D_filters, opt.D_layers)
        setInit(self.Ds, opt.init_type, opt.init_gain)

        self.Dl = Discriminator(2 * self.num_block, opt.D_filters, opt.D_layers)
        setInit(self.Ds, opt.init_type, opt.init_gain)

        # Loss Functions
        self.lambda_L1 = opt.lambda_L1
        self.lambda_local = opt.lambda_local
        self.lambda_style = opt.lambda_style
        self.lambda_content = opt.lambda_content

        self.Lh = Hinge().to(self.device)   # Discriminator Only Loss
        self.Lw = WGANGP().to(self.device)  # Generator Only Loss
        self.L1 = nn.L1Loss()               # Pixel-wise Loss

        # Validation
        self.vL1 = nn.L1Loss()
        self.metrics = metrics

        # Optimizers
        self.G_optimizer = op.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.Ds_optimizer = op.Adam(self.Ds.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.Dc_optimizer = op.Adam(self.Dc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.Dl_optimizer = op.Adam(self.Dc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


    def configure_optimizers(self):
        return [self.G_optimizer, self.Dc_optimizer, self.Ds_optimizer, self.Dl_optimizer], []


    def set_inputs(self, batch):
        self.content = batch.content.to(self.device).view(1, -1, 64, 64)
        self.target = batch.target.to(self.device).view(1, -1, 64, 64)
        self.style = batch.style.to(self.device).view(1, -1, 64, 64)


    def cut_patch(self, fake_images, real_images, true_images):
        batch_size, style_size, H, W = real_images.shape
        fake_block, real_block, true_block = [], [], []

        for b in range(batch_size):
            for i in range(self.num_block):
                style_idx = randint(0, style_size-1)
                x = randint(0, H - self.block_size - 1)
                y = randint(0, W - self.block_size - 1)

                real_random_block = real_images.clone().detach().requires_grad_(False)
                real_random_block = real_random_block[b, style_idx, x:x+self.block_size, y:y+self.block_size].unsqueeze(0)

                fake_random_block = fake_images.clone().detach().requires_grad_(False)
                fake_random_block = fake_random_block[b, 0, x:x+self.block_size, y:y+self.block_size].unsqueeze(0)

                gt_random_block = true_images.clone().detach().requires_grad_(False)
                gt_random_block = gt_random_block[b, 0, x:x+self.block_size, y:y+self.block_size].unsqueeze(0)

                if i == 0:
                    real_blocks = real_random_block
                    fake_blocks = fake_random_block
                    real_blocks = gt_random_block
                else:
                    real_blocks = torch.cat([real_blocks, real_random_block], dim=0)
                    fake_blocks = torch.cat([fake_blocks, fake_random_block], dim=0)
                    real_blocks = torch.cat([real_blocks, gt_random_block], dim=0)

            real_block.append(real_blocks.unsqueeze(0))
            fake_block.append(fake_blocks.unsqueeze(0))
            true_block.append(real_blocks.unsqueeze(0))

        real_block = torch.cat(real_block)
        fake_block = torch.cat(fake_block)
        true_block = torch.cat(true_block)

        return fake_block, real_block, true_block


    def forward(self):
        self.result = self.G(self.content, self.style)

        self.fake_blocks, self.real_blocks, self.true_blocks = \
            self.cut_patch(self.result, self.style, self.target)


    def D_loss(self, real_images, fake_images, discriminator):
        # Discriminator loss on Fake data
        fake = torch.cat(fake_images, 1)
        pred_fake = discriminator(fake.detach())
        loss_D_fake = self.Lh(pred_fake, False)
        # Discriminator loss on Real data
        real = torch.cat(real_images, 1)
        pred_real = discriminator(real)
        loss_D_real = self.Lh(pred_real, True)
        # Combined Loss
        return (loss_D_fake + loss_D_real) * 0.5


    def G_loss(self, fake_images, discriminator):
        # Discriminator loss for Generated data
        fake = torch.cat(fake_images, 1)
        pred_fake = discriminator(fake)
        # Loss according to Discriminator
        return self.Lw(pred_fake, True)


    def D_back(self):
        """Backpropagate Discriminator"""
        self.loss_D_local = self.D_loss([self.real_blocks, self.true_blocks],[self.real_blocks, self.fake_blocks], self.Dl)
        self.loss_D_content = self.D_loss([self.content, self.target],  [self.content, self.result], self.Dc)
        self.loss_D_style = self.D_loss([self.style, self.target], [self.style, self.result], self.Ds)
        # Combined Loss
        self.loss_D = (
            self.loss_D_content * self.lambda_content +
            self.loss_D_style * self.lambda_style +
            self.loss_D_local * self.lambda_local
        )
        # Calculate Gradients
        self.loss_D.backward()


    def G_back(self):
        """Backpropagate Generator"""
        # Discriminator Loss
        self.loss_G_local = self.G_loss([self.real_blocks, self.fake_blocks], self.Dl)
        self.loss_G_content = self.G_loss([self.content, self.result], self.Dc)
        self.loss_G_style = self.G_loss([self.style, self.result], self.Ds)
        self.loss_G_GAN = (
            self.lambda_content * self.loss_G_content +
            self.lambda_style * self.loss_G_style +
            self.lambda_local * self.loss_G_local
        )
        # Generator Loss
        self.loss_G_L1 = self.L1(self.result, self.target)
        # Combined Loss
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 * self.lambda_L1
        # Calculate Gradients
        self.loss_G.backward()


    def training_step(self, batch: TrainBundle, batch_idx: int):
        # Process Batch
        self.set_inputs(batch)
        # Forward Pass
        self.forward()
        # Update Discriminators
        setGrads(True, self.Dc, self.Ds, self.Dl)
        self.Dc_optimizer.zero_grad()
        self.Ds_optimizer.zero_grad()
        self.Dl_optimizer.zero_grad()
        self.D_back()
        self.Dc_optimizer.step()
        self.Ds_optimizer.step()
        self.Dl_optimizer.step()
        # Update Generator
        setGrads(False, self.Dc, self.Ds, self.Dl)
        self.G_optimizer.zero_grad()
        self.G_back()
        self.G_optimizer.step()


    def on_train_epoch_end(self) -> None:
        self.log("Train D Content", self.loss_D_content, on_step=False, on_epoch=True)
        self.log("Train D Style", self.loss_D_style, on_step=False, on_epoch=True)
        self.log("Train G GAN", self.loss_G_GAN, on_step=False, on_epoch=True)
        self.log("Train G L1", self.loss_G_L1, on_step=False, on_epoch=True)


    def validation_step(self, batch: TrainBundle, batch_idx: int):
        # Process Batch
        self.set_inputs(batch)
        # Set to Eval Mode
        self.eval()
        # Forward Pass
        with torch.no_grad():
            self.forward()
        # Compute Loss
        self.val_loss_L1 = self.vL1(self.result, self.target)
        # Compute Metrics
        if self.metrics:
            self.metrics(self.result, self.target)
        # Set to Train Mode
        self.train()


    def on_validation_epoch_end(self) -> None:
        self.log("Valid G L1", self.val_loss_L1, on_step=False, on_epoch=True)
        if self.metrics:
            self.log_dict(self.metrics.compute(), on_step=False, on_epoch=True)
            self.metrics.reset()
