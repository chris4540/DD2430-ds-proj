# from trainer.baseline import ClassificationTrainer


# trainer = ClassificationTrainer(
#     exp_folder="./exp_folders/exp_clsf", log_interval=1, lr=5e-4, eta_min=1e-6, epochs=10)
# trainer.run()
from experiment.catagorical import CatClassification
import numpy as np
exp = CatClassification(
    "./exp_folders/exp_cat",
    lr=0.0005,
    weight_decay=1e-5,
    eta_min=1e-6,
    epochs=20,
    batch_size=128)
exp.run()
