# %%---------------------------------------------------------------------------%

from os import system
from lightning import Trainer
from torch.utils.data import DataLoader

from model.model import Model
from fonts.loader import load
from model.types import TrainBundle, Options
from model.utils import apply, iShow, iShowMany, iSave, iSaveMany

# %%---------------------------------------------------------------------------%

# TODO: data downloading

# %%---------------------------------------------------------------------------%

# Base font is a reference of how symbols should look like.
# It's same for all model training and inference data bundles.
content = load("base").ua

# %%---------------------------------------------------------------------------%

train = load("train")

# %%---------------------------------------------------------------------------%

# train = [*train[:8]]

# %%---------------------------------------------------------------------------%

data = []

for item in train:
    data.extend(
        TrainBundle(
            target=t,
            content=c,
            style=item.en,
            panose=item.panose,
        ) for c, t in zip(content, item.ua)
    )

# %%---------------------------------------------------------------------------%

loader = DataLoader(data, shuffle=True)

# %%---------------------------------------------------------------------------%

opt = Options()

# %%---------------------------------------------------------------------------%

net = Model(opt)

# %%---------------------------------------------------------------------------%

trainer = Trainer(max_epochs=100)

# %%---------------------------------------------------------------------------%

# TODO: advanced checkpointing

trainer.fit(net, loader)

# %%---------------------------------------------------------------------------%

# TODO: loading from checkpoints doesn't work

# net.to_onnx("model.onnx", X)
# trainer.save_checkpoint("model.ckpt")
# net.load_from_checkpoint("model.ckpt", opt=opt)
# system("shutdown /s /t 1")

# %%---------------------------------------------------------------------------%

X = loader.dataset[25]
iShowMany(X.content, X.target, apply(net, X))

# %%---------------------------------------------------------------------------%

x = apply(net, X)
iSave(x)

# %%---------------------------------------------------------------------------%