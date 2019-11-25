from trainer.baseline import ClassificationTrainer


trainer = ClassificationTrainer(
    exp_folder="./exp_folders/exp_clsf", log_interval=5, lr=1e-2, epochs=10)
trainer.run()
