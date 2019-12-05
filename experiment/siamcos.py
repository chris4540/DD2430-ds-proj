class SiameseCosDistance:
    """
    Example:
    >> exp = SiameseCosDistance()
    >> exp.run(max_epochs=10)
    """
    def __init__(self):
        pass


    #
    # Exp definition
    #
    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def hparams(self):
        return self._hparams


    def run(self, max_epochs=10):
        pass
