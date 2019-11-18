from abc import ABC as AbstractBaseClass
from abc import abstractmethod

from tqdm import tqdm


class TrainingAbstractMethod(AbstractBaseClass):

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


class BaseTrainingMethod(AbstractBaseClass):

    # for callback use

    log_cfg = {
        'desc': "ITERATION - loss: {:.2f}",
        'pbar': None,
        'train_loader_len': -1,
        'interval': 50,
    }

    def __init__(self, log_interval):
        self.log_cfg['interval'] = log_interval

    def prepare_before_run(self):
        self.prepare_data_loaders()
        self.prepare_exp_settings()
        self.prepare_logging()

    def prepare_logging(self):

        # Try to get train_loader_len
        if hasattr(self, 'train_loader'):
            train_loader_len = len(self.train_loader)
        elif hasattr(self, 'train_loader_len'):
            train_loader_len = self.train_loader_len
        else:
            raise RuntimeError(
                "Unable to determine the length of train loader.")

        #
        self.log_cfg['train_loader_len'] = train_loader_len
        self.log_cfg['pbar'] = tqdm(
            initial=0, leave=False, total=train_loader_len,
            desc=self.log_cfg['desc'].format(0)
        )

    # Pre-defined callbacks
    def take_scheduler_step(self, engine):
        self.scheduler.step()
        # Print out
        tqdm.write("Learning Rate - Epoch: {}  Learning Rate: {}"
                   .format(engine.state.epoch, self.scheduler.get_lr()))

    def log_training_loss(self, engine):
        train_loader_len = self.log_cfg['train_loader_len']
        log_interval = self.log_cfg['interval']
        # -------------------------------
        n_iter = (engine.state.iteration - 1) % train_loader_len + 1

        if n_iter % log_interval == 0:
            desc = self.log_cfg['desc']
            pbar = self.log_cfg['pbar']
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)

    def log_training_results(self, engine, train_loader, evaluator):
        pbar = self.log_cfg['pbar']
        # -----------------------------------------------
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_loss)
        )
