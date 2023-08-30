# %%---------------------------------------------------------------------------%

from os import system
from pickle import load

from lightning import Trainer
from torch.utils.data import DataLoader

from model.model import Model
from model.types import TrainBundle, Options
from fonts.panose1 import features_to_digits
from model.utils import apply, show, showMany

# %%---------------------------------------------------------------------------%

with open("-data/base.pkl", "rb") as X:
    base = load(X)

with open("-data/train.pkl", "rb") as X:
    train = load(X)

with open("-data/test.pkl", "rb") as X:
    test = load(X)

# %%---------------------------------------------------------------------------%

# Base font as a reference of how symbols should look like
# It's same for all model training data bundles
content = base.cyr

# %%---------------------------------------------------------------------------%

train = [*train[50:51]]

# %%---------------------------------------------------------------------------%

data = []

for item in train:
    # TODO: store panose as number array in DB
    panose = features_to_digits(item.panose)

    data.extend(
        TrainBundle(
            target=t,
            content=c,
            panose=panose,
            style=item.lat,
        ) for c, t in zip(content, item.cyr)
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
showMany(X.content, X.target, apply(net, X))

# %%---------------------------------------------------------------------------%