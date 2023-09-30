from dataclasses import dataclass


@dataclass
class Options:
    """Model configuration."""
    # *------------------------------------------------------------------------* Training & Optimization Options
    # network initialization [normal | xavier | kaiming | orthogonal]
    init_type: str = "normal"
    # scaling factor for initializer
    init_gain: float = 0.02
    # momentum term of adam
    beta1: float = 0.5
    # initial learning rate
    lr: float = 0.0002

    # *------------------------------------------------------------------------* Model Options
    # number of reference style images
    refs: int = 52
    # number of G filters in the last conv layer
    G_filters: int = 64
    # number of D filters in the first conv layer
    D_filters: int = 64
    # dropout usage for the generator
    G_dropout: bool = False
    # number of discriminator blocks
    D_layers: int = 3
    # GAN objective function: { vanilla, lsgan, wgangp, hinge }
    objective: str = "hinge"
    # weight for content loss
    lambda_content: float = 1.0
    # weight for style loss
    lambda_style: float = 1.0
    # weight for L1 loss
    lambda_L1: float = 100.0

    # *------------------------------------------------------------------------* GAS-NeXt Specific Options
    # weight for local loss
    lambda_local: float = 0.1
    # size of GAS-NeXt blocks
    block_size: int = 32
    # number of GAS-NeXt blocks
    num_block: int = 7