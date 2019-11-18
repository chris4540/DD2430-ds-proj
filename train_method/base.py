from abc import ABC as AbstractBaseClass
from abc import abstractmethod

class TrainingAbstractMethod(AbstractBaseClass):

    @abstractmethod
    def run(self):
        pass
