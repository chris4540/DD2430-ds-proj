import numpy as np
from experiment.siamcos import SiameseCosDistanceWithCat

margin = np.round(np.sqrt(0.2), decimals=4)
exp = SiameseCosDistanceWithCat(
    "./exp_folders/exp_siamcos",
    lr=0.0005,
    weight_decay=1e-5,
    eta_min=1e-6,
    epochs=20,
    batch_size=128,
    margin=margin)

exp.run()
