class Options:
    def __init__(self):
        # *--------------------------------------------------------------------* Training & Optimization Options
        # network initialization [normal | xavier | kaiming | orthogonal]
        self.init_type: str = 'normal'
        # scaling factor for normal, xavier and orthogonal.
        self.init_gain: float = 0.02
        # momentum term of adam
        self.beta1: float = 0.5
        # initial learning rate for adam
        self.lr: float = 0.0002

        # *--------------------------------------------------------------------* Model Options
        # number of reference style images
        self.refs: int = 5
        # number of G filters in the last conv layer
        self.ngf: int = 64
        # number of D filters in the first conv layer
        self.ndf: int = 64
        # number of discriminator blocks
        self.D_layers: int = 3
        # use dropout for the generator
        self.G_dropout: bool = False
        # type of GAN objective. [vanilla| lsgan | wgangp | hinge]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.
        self.gan_mode: str = 'hinge'
        # weight for content loss
        self.lambda_content = 1.0
        # weight for style loss
        self.lambda_style = 1.0
        # weight for L1 loss
        self.lambda_L1 = 100.0