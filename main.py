# %%---------------------------------------------------------------------------%

import pandas as pd

from torch import Tensor
from lightning import Trainer
from torch.utils.data import DataLoader

from model.model import Model
from model.types import TrainBundle, Options
from model.utils import apply, show, showMany

# %%---------------------------------------------------------------------------%

raw_os_en = pd.read_csv("model/data/OS-EN.csv", header=None)
raw_os_ua = pd.read_csv("model/data/OS-UA.csv", header=None)
raw_if_en = pd.read_csv("model/data/IF-EN.csv", header=None)

raw_os_en.drop(0, axis=1, inplace=True)
raw_os_ua.drop(0, axis=1, inplace=True)
raw_if_en.drop(0, axis=1, inplace=True)

# %%---------------------------------------------------------------------------%

if_en = Tensor(raw_if_en.values.reshape(-1, 64, 64))
os_en = Tensor(raw_os_en.values.reshape(-1, 64, 64))
os_ua = Tensor(raw_os_ua.values.reshape(-1, 64, 64))

X = if_en
S = os_ua
Y = os_en

# %%---------------------------------------------------------------------------%

data = [TrainBundle(content=x, target=y, style=S[:5]) for x, y in zip(X, Y)]

loader = DataLoader(data, batch_size=1)

# %%---------------------------------------------------------------------------%

opt = Options()

# %%---------------------------------------------------------------------------%

net = Model(opt)

# %%---------------------------------------------------------------------------%

trainer = Trainer(max_epochs=100)

# %%---------------------------------------------------------------------------%

trainer.fit(net, loader)

# %%---------------------------------------------------------------------------%

X = loader.dataset[0]
showMany(X.content, X.target)

# %%---------------------------------------------------------------------------%

Y = apply(net, X)

# %%---------------------------------------------------------------------------%

show(Y)

# %%---------------------------------------------------------------------------%
