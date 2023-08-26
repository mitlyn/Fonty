from lightning import LightningModule
import torch, torch.nn as nn, torch.optim as op

from model.blocks import Generator, Discriminator, GANLoss
from model.types import TrainBundle, Options
from model.utils import setInit

# TODO: callbacks, learning rate schedulers
# TODO: metrics

# *----------------------------------------------------------------------------*


class Model(LightningModule):
    def __init__(self, opt: Options):
        super(Model, self).__init__()

        self.automatic_optimization = False

        self.G = Generator(opt.G_filters, blocks=6, dropout=opt.G_dropout)
        setInit(self.G, opt.init_type, opt.init_gain)

        self.Dc = Discriminator(2, opt.D_filters, opt.D_layers)
        setInit(self.Dc, opt.init_type, opt.init_gain)

        self.Ds = Discriminator(opt.refs + 1, opt.D_filters, opt.D_layers)
        setInit(self.Ds, opt.init_type, opt.init_gain)

        # Loss Functions
        self.lambda_L1 = opt.lambda_L1
        self.lambda_style = opt.lambda_style
        self.lambda_content = opt.lambda_content

        self.L1 = nn.L1Loss()
        self.LG = GANLoss(opt.gan_mode).to(self.device)

        # Optimizers
        self.G_optimizer = op.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.Ds_optimizer = op.Adam(self.Ds.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.Dc_optimizer = op.Adam(self.Dc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


    def toggle_grads(self, state: bool, *nets):
        for net in nets:
            for param in net.parameters():
                param.requires_grad = state


    def configure_optimizers(self):
        return [self.G_optimizer, self.Dc_optimizer, self.Ds_optimizer], []


    def forward(self):
        self.result = self.G((self.content, self.style))


    def D_loss(self, real_images, fake_images, discriminator):
        # Fake Loss
        fake = torch.cat(fake_images, 1)
        pred_fake = discriminator(fake.detach())
        loss_D_fake = self.LG(pred_fake, False)
        # Real Loss
        real = torch.cat(real_images, 1)
        pred_real = discriminator(real)
        loss_D_real = self.LG(pred_real, True)
        # Combined Loss
        return (loss_D_fake + loss_D_real) * 0.5


    def G_loss(self, fake_images, discriminator):
        fake = torch.cat(fake_images, 1)
        pred_fake = discriminator(fake)
        # Generator Loss
        return self.LG(pred_fake, True, True)


    def D_back(self):
        """Calculate Discriminator loss"""
        self.loss_D_content = self.D_loss([self.content, self.target],  [self.content, self.result], self.Dc)
        self.loss_D_style = self.D_loss([self.style, self.target], [self.style, self.result], self.Ds)
        self.loss_D = self.loss_D_content * self.lambda_content + self.loss_D_style * self.lambda_style

        self.loss_D.backward()


    def G_back(self):
        """Calculate Generator loss (L1 & GAN)"""
        # First, G(A) should fake the discriminator
        self.loss_G_content = self.G_loss([self.content, self.result], self.Dc)
        self.loss_G_style = self.G_loss([self.style, self.result], self.Ds)
        self.loss_G_GAN = self.lambda_content * self.loss_G_content + self.lambda_style * self.loss_G_style

        # Second, G(A) = B
        self.loss_G_L1 = self.L1(self.result, self.target)
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 * self.lambda_L1

        self.loss_G.backward()


    def training_step(self, batch: TrainBundle, batch_idx: int):
        # Process Batch
        self.content = batch.content.to(self.device).view(1, -1, 64, 64)
        self.target = batch.target.to(self.device).view(1, -1, 64, 64)
        self.style = batch.style.to(self.device).view(1, -1, 64, 64)

        # Forward Pass
        self.forward()

        # Update Discriminators
        self.toggle_grads(True, self.Dc, self.Ds)
        self.Dc_optimizer.zero_grad()
        self.Ds_optimizer.zero_grad()
        self.D_back()
        self.Dc_optimizer.step()
        self.Ds_optimizer.step()

        # Update Generator
        self.toggle_grads(False, self.Dc, self.Ds)
        self.G_optimizer.zero_grad()
        self.G_back()
        self.G_optimizer.step()


    def on_train_epoch_end(self) -> None:
        self.log("loss_D_content", self.loss_D_content, on_step=False, on_epoch=True)
        self.log("loss_D_style", self.loss_D_style, on_step=False, on_epoch=True)
        self.log("loss_G_GAN", self.loss_G_GAN, on_step=False, on_epoch=True)
        self.log("loss_G_L1", self.loss_G_L1, on_step=False, on_epoch=True)
