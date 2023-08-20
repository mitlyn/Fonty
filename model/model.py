import torch
import lightning as L
import torch.nn as nn
import torch.optim as op

from model.blocks import Generator, Discriminator, GANLoss
from model.utils import Options, setInit

# TODO: callbacks, learning rate schedulers
# TODO: metrics

# *----------------------------------------------------------------------------*


class Model(L.LightningModule):
    def __init__(self, opt: Options):
        super(Model, self).__init__()

        self.automatic_optimization = False

        self.G = Generator(opt.ngf, n_blocks=6, dropout=opt.G_dropout)
        setInit(self.G, opt.init_type, opt.init_gain)

        self.Dc = Discriminator(2, opt.ndf, opt.D_layers)
        setInit(self.Dc, opt.init_type, opt.init_gain)

        self.Ds = Discriminator(opt.refs + 1, opt.ndf, opt.D_layers)
        setInit(self.Ds, opt.init_type, opt.init_gain)

        # Loss Functions
        self.lambda_L1 = opt.lambda_L1
        self.lambda_style = opt.lambda_style
        self.lambda_content = opt.lambda_content

        self.L1 = nn.L1Loss()
        self.LG = GANLoss(opt.gan_mode).to(self.device)

        # Optimizers
        self.optimizer_G = op.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D_style = op.Adam(self.Ds.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D_content = op.Adam(self.Dc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


    def set_input(self, data):
        self.style_images = data['style'].to(self.device).view(1, -1, 64, 64)
        self.target_images = data['target'].to(self.device).view(1, -1, 64, 64)
        self.content_images = data['content'].to(self.device).view(1, -1, 64, 64)


    def toggle_grads(self, state: bool, *nets):
        for net in nets:
            for param in net.parameters():
                param.requires_grad = state


    def configure_optimizers(self):
        return [self.optimizer_G, self.optimizer_D_content, self.optimizer_D_style], []


    def forward(self):
        self.result = self.G((self.content_images, self.style_images))


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
        self.loss_D_content = self.D_loss([self.content_images, self.target_images],  [self.content_images, self.result], self.Dc)
        self.loss_D_style = self.D_loss([self.style_images, self.target_images], [self.style_images, self.result], self.Ds)
        self.loss_D = self.loss_D_content * self.lambda_content + self.loss_D_style * self.lambda_style

        self.loss_D.backward()


    def G_back(self):
        """Calculate Generator loss (L1 & GAN)"""
        # First, G(A) should fake the discriminator
        self.loss_G_content = self.G_loss([self.content_images, self.result], self.Dc)
        self.loss_G_style = self.G_loss([self.style_images, self.result], self.Ds)
        self.loss_G_GAN = self.lambda_content * self.loss_G_content + self.lambda_style * self.loss_G_style

        # Second, G(A) = B
        self.loss_G_L1 = self.L1(self.result, self.target_images)
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 * self.lambda_L1

        self.loss_G.backward()


    def training_step(self, batch, batch_idx):
        self.set_input(batch)
        self.forward()

        # Update D
        self.toggle_grads(True, self.Dc, self.Ds)
        self.optimizer_D_content.zero_grad()
        self.optimizer_D_style.zero_grad()
        self.D_back()
        self.optimizer_D_content.step()
        self.optimizer_D_style.step()

        # Update G
        self.toggle_grads(False, self.Dc, self.Ds)
        self.optimizer_G.zero_grad()
        self.G_back()
        self.optimizer_G.step()


    def on_train_epoch_end(self) -> None:
        self.log("loss_D_content", self.loss_D_content, on_step=False, on_epoch=True)
        self.log("loss_D_style", self.loss_D_style, on_step=False, on_epoch=True)
        self.log("loss_G_GAN", self.loss_G_GAN, on_step=False, on_epoch=True)
        self.log("loss_G_L1", self.loss_G_L1, on_step=False, on_epoch=True)
