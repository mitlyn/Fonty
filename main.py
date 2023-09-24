# %%---------------------------------------------------------------------------%

import pickle as pkl
from os import system
from random import shuffle
from asq import query as Q
from seaborn import heatmap
from lightning import Trainer
from torch.onnx import export as onnx
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch import save as saveModel, load as loadModel
from torchmetrics import MetricCollection, image as mi

from model.emd import EMD
from model.fonty import Fonty
from fonts.loader import load
from model.types import Options
from model.utils import apply, bundled, iShow, iShowMany, iSave, iSaveMany, i2Save, i3Save, is3Save

# TODO: FID, LPIPS
# TODO: best discriminator (trained separately)
# TODO: interchangeable generators
# TODO: forward((..)) for easier inference

# %%---------------------------------------------------------------------------% Loading Data

# Base font is a reference of how symbols should look like.
# It's same for all model training and inference data bundles.
content = load("base").ua

# %%---------------------------------------------------------------------------% Font Selection

# fonts = load("train")
# shuffle(fonts)
# pick = fonts

# pick = [*fonts[:10]]

# pick = []

# pick += (
#     Q(fonts)
#         .where(lambda x: "Regular" in x.name)
#         .where(lambda x: x.name not in [
#             'Lobster Regular',
#             'Pacifico Regular',
#             'Amatic SC Regular',
#             'Rubik Iso Regular',
#             'Montserrat Alternates Regular'
#         ]).to_list()
# )

# Q(pick).select(lambda x: iShowMany(x.ua[0])).to_tuple()

# pick += Q(fonts).where(lambda x: "Alice" in x.name).to_list()
# pick += Q(fonts).where(lambda x: "Times" in x.name).to_list()
# pick += Q(fonts).where(lambda x: "Spectral" in x.name).to_list()

# %%---------------------------------------------------------------------------% Train/Valid Split + Cache

# train_fonts, valid_fonts = train_test_split(pick, test_size=0.2)

# pkl.dump(train_fonts, open("data/train_fonts.pkl", "wb"))
# pkl.dump(valid_fonts, open("data/valid_fonts.pkl", "wb"))

train_fonts = pkl.load(open("data/train_fonts.pkl", "rb"))
valid_fonts = pkl.load(open("data/valid_fonts.pkl", "rb"))

# %%---------------------------------------------------------------------------% Bundles & Loaders

train_data = bundled(train_fonts, content, train=True)
valid_data = bundled(valid_fonts, content, train=True)

train_loader = DataLoader(train_data, shuffle=True)
valid_loader = DataLoader(valid_data, shuffle=False)

# %%---------------------------------------------------------------------------% Model Setup

opt = Options()

metrics = MetricCollection({
    "PSNR": mi.PeakSignalNoiseRatio(),
    "RASE": mi.RelativeAverageSpectralError(),
    "SSIM": mi.StructuralSimilarityIndexMeasure(),
    "RMSE-SW": mi.RootMeanSquaredErrorUsingSlidingWindow(),
    "ERGAS": mi.ErrorRelativeGlobalDimensionlessSynthesis(),
})

model = EMD(opt, metrics)

# %%---------------------------------------------------------------------------% Trainer Setup

trainer = Trainer(max_epochs=100)

# %%---------------------------------------------------------------------------% Training

trainer.fit(model, train_loader, valid_loader)

# %%---------------------------------------------------------------------------%

# saveModel(model, "small.pt")
# model = loadModel("model.pt")

# %%---------------------------------------------------------------------------% Result Inspection

# fake = Q(train_loader.dataset).select(lambda x: x.target).to_list()
# real = Q(train_loader.dataset).select(lambda x: x.target).to_list()

for i in range(10):
    X = train_loader.dataset[i]
    iShowMany(X.content, X.target, apply(model, X))

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

# x = apply(model, X)
# iSave(x, "test")

# %%---------------------------------------------------------------------------%

W = model.G.Ep.linear[0].weight.detach().cpu().numpy()
heatmap(W, cmap="coolwarm")

# %%---------------------------------------------------------------------------%

# for i in range(len(train_loader.dataset)):
#     X = train_loader.dataset[i]
#     Y = apply(model, X)
#     i3Save(X.content, X.style, Y, f"out3/{i}")

# %%---------------------------------------------------------------------------%

# onnx(model.G.Ep.to("cuda"), {"panose":model.panose.to("cuda")}, "G.onnx")

# %%---------------------------------------------------------------------------%