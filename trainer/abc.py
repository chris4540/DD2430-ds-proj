"""
Abstract training class
"""
from abc import ABC as AbstractBaseClass
from abc import abstractmethod


class AdstractTrainer(AbstractBaseClass):

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def prepare_data_loaders(self):
        """
        For preparing data loaders and save them as instance attributes
        """
        pass

    @abstractmethod
    def prepare_exp_settings(self):
        """
        Define stuff which are before the actual run. For example:
            - Optimizer
            - Model
        """
        pass

    @abstractmethod
    def prepare_logging(self):
        pass
