from train_method.siamese import SiameseFashionMNIST


hyper_params = dict()
experiment = SiameseFashionMNIST(hyper_params)
experiment.run()
