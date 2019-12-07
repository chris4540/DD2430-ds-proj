import numpy as np
from experiment.siamcat import SiameseEucDistanceWithCat

for m_sq in [1, 10, 100]:
    margin = np.round(np.sqrt(m_sq), decimals=4)
    exp = SiameseEucDistanceWithCat(
        "./exp_folders/exp_siamcat_m2_{}".format(m_sq),
        lr=5e-4,
        weight_decay=1e-5,
        eta_min=1e-6,
        batch_size=128,
        epochs=20,
        lambda_=1,
        margin=margin)
    exp.run()
