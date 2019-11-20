from trainer.siamese import SiameseTrainer


# hyper_params = dict()
experiment = SiameseTrainer(log_interval=5, lr=1e-2, epochs=2)
experiment.run()
