# %%---------------------------------------------------------------------------%

from pickle import load

from lightning import Trainer
from torch.utils.data import DataLoader

from model.model import Model
from model.types import TrainBundle, Options
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

# Drop first element of train data
# Fucking Jura Light broke all of my hopes
train = train[1:]

# %%---------------------------------------------------------------------------%



# %%---------------------------------------------------------------------------%

data = []

for i in range(len(train)):
    data.extend([
        TrainBundle(
            content=c,
            target=t,
            panose=t.panose,
            style=train[i].lat,
        ) for c, t in zip(content, train[i].cyr)
    ])

# %%---------------------------------------------------------------------------%

loader = DataLoader(data, batch_size=1, shuffle=True)

# %%---------------------------------------------------------------------------%

opt = Options(refs=52)

# %%---------------------------------------------------------------------------%

net = Model(opt)

# %%---------------------------------------------------------------------------%

trainer = Trainer(max_epochs=5)

# %%---------------------------------------------------------------------------%

trainer.fit(net, loader)

# %%---------------------------------------------------------------------------%

trainer.save_checkpoint("model.ckpt")

# %%---------------------------------------------------------------------------%

X = loader.dataset[2999]
showMany(X.content, X.target)

# %%---------------------------------------------------------------------------%

Y = apply(net, X)

# %%---------------------------------------------------------------------------%

show(Y)

# %%---------------------------------------------------------------------------%
