"""
Do robust searching
"""

import numpy as np
from trainer.baseline import ClassificationTrainer


def find_lr():
    log10_lrs = np.random.uniform(-4, -1.5, size=10)
    lrs = np.power(10, log10_lrs)

    for lr in lrs:
        trainer = ClassificationTrainer(
            exp_folder="./exp_folders/exp_clsf", log_interval=1, lr=lr, epochs=1, batch_size=100)
        trainer.run()
