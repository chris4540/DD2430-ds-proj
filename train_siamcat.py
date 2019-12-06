import numpy as np
from experiment.siamcat import SiameseEucDistanceWithCat

m_sq = 1
lambdas = {
    1000: 4.0,
    100: 2.0,
    10: 1.0,
    1: 1,
}
margin = np.round(np.sqrt(m_sq), decimals=4)
exp = SiameseEucDistanceWithCat(
    "./exp_folders/exp_siamcat_m2_{}".format(m_sq),
    lr=5e-4,
    weight_decay=1e-5,
    eta_min=1e-6,
    batch_size=128,
    epochs=10,
    lambda_=lambdas[m_sq],
    margin=margin)

exp.run()
