from trainer.baseline import BaselineFashionMNIST


experiment = BaselineFashionMNIST(log_interval=5, lr=1e-2, epochs=2)
experiment.run()
