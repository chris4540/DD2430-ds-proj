import numpy as np
from experiment.siamcat import SiameseEucDistanceWithCat

margin = np.round(np.sqrt(1), decimals=4)
exp = SiameseEucDistanceWithCat(
    "./exp_folders/exp_siameducat",
    debug=True,
    lr=5e-4,
    weight_decay=1e-5,
    eta_min=1e-6,
    batch_size=128,
    epochs=10,
    margin=margin)

exp.run()
