# %%---------------------------------------------------------------------------%

import pickle as pkl
from os import system
from asq import query as Q
from random import shuffle
from seaborn import heatmap
from pprint import pprint as pp
from torch.onnx import export as onnx
from sklearn.model_selection import train_test_split
from torch import save as saveModel, load as loadModel

from lightning import Trainer
from torch.utils.data import DataLoader

from model.ftransgan import FTransGAN
from model.gasnext import GasNeXt
from model.fonty import Fonty
from model.emd import EMD

from fonts.loader import load
from model.types import Options
from model.utils.eval import Metrics, eval, evalPanose
from model.utils.nets import bundled, apply, applyPanose
from model.utils.show import iShow, iShowMany, iSave, iSaveMany, i2Save, i3Save, is3Save

# TODO: frozen universal discriminator (trained separately)

# %%---------------------------------------------------------------------------% Loading Data

# Base font is a reference of how symbols should look like.
# It's same for all model training and inference data bundles.
content = load("base").ua

# %%---------------------------------------------------------------------------% Font Selection

# Load all fonts from the dataset.
# Train fonts have Cyrillic symbols defined and are trained on.
# Test fonts have only Latin symbols and are used for inference.
fonts = load("train")

# %%---------------------------------------------------------------------------% Train/Valid Split + Cache

train_fonts, valid_fonts = train_test_split(fonts, test_size=0.2)

# pkl.dump(train_fonts, open("data/train_fonts.pkl", "wb"))
# pkl.dump(valid_fonts, open("data/valid_fonts.pkl", "wb"))

# train_fonts = pkl.load(open("data/train_fonts.pkl", "rb"))
# valid_fonts = pkl.load(open("data/valid_fonts.pkl", "rb"))

# %%---------------------------------------------------------------------------% Bundles & Loaders

train_data = bundled(train_fonts, content, train=True)
valid_data = bundled(valid_fonts, content, train=True)

train_loader = DataLoader(train_data, shuffle=True)
valid_loader = DataLoader(valid_data, shuffle=False)

# %%---------------------------------------------------------------------------% Model Setup

options = Options()
metrics = Metrics()

model = Fonty(options, metrics)

# %%---------------------------------------------------------------------------% Trainer Setup

trainer = Trainer(max_epochs=1000)

# %%---------------------------------------------------------------------------% Training

trainer.fit(model, train_loader, valid_loader)

# %%---------------------------------------------------------------------------%

saveModel(model.cpu(), "model.pt")
model = loadModel("model.pt").cuda()

# %%---------------------------------------------------------------------------% Result Inspection

train_score = evalPanose(model, train_loader.dataset)

for k in train_score:
    print(f"{k}: {train_score[k]}")

# %%---------------------------------------------------------------------------% Result Illustration

valid_score = evalPanose(model, valid_loader.dataset)

for k in valid_score:
    print(f"{k}: {valid_score[k]}")

# %%---------------------------------------------------------------------------%

X = train_loader.dataset[1272]
iShowMany(X.content, X.target, applyPanose(model, X))

# %%---------------------------------------------------------------------------% Result Illustration

# # Good: 273, 333, 291, 50, 70
# # Bad: 275, 791, 751, 121, 514

# good = [273, 333, 291, 751, 70]
# fail = [275, 791, 50, 121, 514]

# R1 = Q(good).select(lambda x: train_loader.dataset[x]).select(lambda I: {"X": I.content, "S": I.style[0], "Y": apply(model, I)}).to_list()
# R2 = Q(fail).select(lambda x: train_loader.dataset[x]).select(lambda I: {"X": I.content, "S": I.style[0], "Y": apply(model, I)}).to_list()

# # X = train_loader.dataset[514]
# # iShowMany(X.content, X.target, apply(model, X))
# # i3Save(X.content, X.style[0], apply(model, X), "test")
# is3Save(R1, "good")
# is3Save(R2, "fail")

# %%---------------------------------------------------------------------------%

W = model.G.Ep.linear[0].weight.detach().cpu().numpy()
heatmap(W, cmap="coolwarm")

# %%---------------------------------------------------------------------------%

for i in range(len(train_loader.dataset)):
    X = train_loader.dataset[i]
    Y = apply(model, X)
    i3Save(X.content, X.style, Y, f"out/{i}")

# %%---------------------------------------------------------------------------%

# onnx(model.G.Ep.to("cuda"), {"panose":model.panose.to("cuda")}, "Fonty.onnx")

# %%---------------------------------------------------------------------------%