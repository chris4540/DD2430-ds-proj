from trainer.siamese import SiameseFashionMNISTTrainer


# hyper_params = dict()
experiment = SiameseFashionMNISTTrainer(log_interval=5, lr=1e-2, epochs=2)
experiment.run()
