# %%---------------------------------------------------------------------------%

from os import system
from random import shuffle
from seaborn import heatmap
from lightning import Trainer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch import save as saveModel, load as loadModel
from torchmetrics import MetricCollection, image as mi

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
opt.G_dropout = 0.1

metrics = MetricCollection({
    "PSNR": mi.PeakSignalNoiseRatio(),
    "RASE": mi.RelativeAverageSpectralError(),
    "SSIM": mi.StructuralSimilarityIndexMeasure(),
    "RMSE-SW": mi.RootMeanSquaredErrorUsingSlidingWindow(),
    "ERGAS": mi.ErrorRelativeGlobalDimensionlessSynthesis(),
})

# %%---------------------------------------------------------------------------%

model = Model(opt, metrics)

# %%---------------------------------------------------------------------------%

trainer = Trainer(max_epochs=100)

# %%---------------------------------------------------------------------------%

trainer.fit(model, train_loader, valid_loader)

# %%---------------------------------------------------------------------------%

saveModel(model, "small.pt")
# model = loadModel("model.pt")

# net.to_onnx("model.onnx", X)
# trainer.save_checkpoint("model.ckpt")
# net.load_from_checkpoint("model.ckpt", opt=opt)
# system("shutdown /s /t 1")

# %%---------------------------------------------------------------------------%

X = valid_loader.dataset[2]
iShowMany(X.content, X.target, apply(model, X))

# for i in range(26):
#     X = valid_loader.dataset[i]
#     iShowMany(X.content, X.target, apply(model, X))

# %%---------------------------------------------------------------------------%

x = apply(model, X)
iSave(x, "test")

# %%---------------------------------------------------------------------------%