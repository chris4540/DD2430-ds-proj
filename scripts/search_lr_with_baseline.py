"""
Do robust searching
--------- Hyper Parameters ---------
batch_size                     100
epochs                         1
eta_min                        1e-05
lr                             0.0006878505308482407
--------- Hyper Parameters ---------
Training Results - Epoch: 1  Avg accuracy: 0.69 Avg loss: 0.98
Validation Results - Epoch: 1  Avg accuracy: 0.65 Avg loss: 1.10
--------- Hyper Parameters ---------
batch_size                     100
epochs                         1
eta_min                        1e-05
lr                             0.00022780192919486867
--------- Hyper Parameters ---------
Training Results - Epoch: 1  Avg accuracy: 0.71 Avg loss: 0.94
Validation Results - Epoch: 1  Avg accuracy: 0.65 Avg loss: 1.08
"""
import os.path
import sys
cur_path = os.path.realpath(__file__)
cur_dir = os.path.dirname(cur_path)
parent_dir = cur_dir[:cur_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
import numpy as np
from trainer.baseline import ClassificationTrainer


def find_lr():
    log10_lrs = np.random.uniform(-4, -1.5, size=10)
    lrs = np.power(10, log10_lrs)

    for lr in lrs:
        trainer = ClassificationTrainer(
            exp_folder="./exp_folders/exp_clsf", log_interval=1, lr=lr, epochs=1, batch_size=100)
        trainer.run()


if __name__ == "__main__":
    find_lr()
