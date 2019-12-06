import numpy as np
from experiment.siamcos import SiameseCosDistanceWithCat

margin = np.round(np.sqrt(0.2), decimals=4)
exp = SiameseCosDistanceWithCat(
    "./exp_folders/exp_siamcos",
    debug=True,
    lr=5e-4, weight_decay=1e-5, eta_min=1e-6, batch_size=128, margin=margin)

exp.run()
