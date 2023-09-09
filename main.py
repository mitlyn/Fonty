# %%---------------------------------------------------------------------------%

from os import system
from random import shuffle
from lightning import Trainer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from fonts.loader import load
from model.model import Model
from model.types import Options
from model.utils import apply, bundled, iShow, iShowMany, iSave, iSaveMany

# %%---------------------------------------------------------------------------%

# TODO: data downloading

# %%---------------------------------------------------------------------------%

# Base font is a reference of how symbols should look like.
# It's same for all model training and inference data bundles.
content = load("base").ua

# %%---------------------------------------------------------------------------%

fonts = load("train")
shuffle(fonts)

# fonts = [*fonts[:10]]

# %%---------------------------------------------------------------------------%

train_fonts, valid_fonts = train_test_split(fonts, test_size=0.2)

# %%---------------------------------------------------------------------------%

train_data = bundled(train_fonts, content, train=True)
valid_data = bundled(valid_fonts, content, train=True)

# %%---------------------------------------------------------------------------%

train_loader = DataLoader(train_data, shuffle=True)
valid_loader = DataLoader(valid_data, shuffle=True)

# %%---------------------------------------------------------------------------%

opt = Options()

# %%---------------------------------------------------------------------------%

model = Model(opt)

# %%---------------------------------------------------------------------------%

trainer = Trainer(max_epochs=100)

# %%---------------------------------------------------------------------------%

# TODO: advanced checkpointing

trainer.fit(model, train_loader, valid_loader)

# %%---------------------------------------------------------------------------%

# TODO: loading from checkpoints doesn't work

# net.to_onnx("model.onnx", X)
# trainer.save_checkpoint("model.ckpt")
# net.load_from_checkpoint("model.ckpt", opt=opt)
# system("shutdown /s /t 1")

# %%---------------------------------------------------------------------------%

X = train_loader.dataset[25]
iShowMany(X.content, X.target, apply(model, X))

# %%---------------------------------------------------------------------------%

x = apply(model, X)
iSave(x, "test")

# %%---------------------------------------------------------------------------%