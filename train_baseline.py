from trainer.baseline import ClassificationTrainer


trainer = ClassificationTrainer(
    exp_folder="./exp_folders/exp_clsf", log_interval=1, lr=5e-4, eta_min=1e-6, epochs=10)
trainer.run()
