from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from .abc import AdstractTrainer


class BaseTrainer(AdstractTrainer):
    """
    Basic training method which has a lot of common display settings and callbacks
    """

    # for callback use

    log_cfg = {
        'desc': "ITERATION - loss: {:.2f}",
        'pbar': None,
        'train_loader_len': -1,
        'interval': 50,
    }

    def __init__(self, exp_folder, log_interval):
        if not isinstance(log_interval, int) or not log_interval > 0:
            raise ValueError("log_interval is a positive integer.")
        self.log_cfg['interval'] = log_interval
        # ------------------------------
        # Setup experiement folder

        # Move it if folder exists
        folder = Path(exp_folder)
        if folder.exists():
            folder_suffix = datetime.utcnow().strftime("%y%m%d_%H%M%S")
            folder.rename(str(folder) + '_' + folder_suffix)

        # mkdir -p <exp_folder>
        self.exp_folder = Path(exp_folder)
        self.exp_folder.mkdir()

    def prepare_before_run(self):
        self.prepare_data_loaders()
        self.prepare_exp_settings()
        self.prepare_logging()

    def prepare_logging(self):

        # Try to get train_loader_len
        if hasattr(self, 'train_loader_len'):
            train_loader_len = self.train_loader_len
        elif hasattr(self, 'train_loader'):
            train_loader_len = len(self.train_loader)
        else:
            raise RuntimeError(
                "Unable to determine the length of train loader.")

        #
        self.log_cfg['train_loader_len'] = train_loader_len
        self.log_cfg['pbar'] = tqdm(
            initial=0, leave=False, total=train_loader_len,
            desc=self.log_cfg['desc'].format(0)
        )
    # ------------------------
    # Pre-defined callbacks
    # ------------------------
    # overide and register to ignite engine if needed

    def take_scheduler_step(self, engine):
        self.scheduler.step()
        # Print out
        tqdm.write("Learning Rate - Epoch: {}  Learning Rate: {}"
                   .format(engine.state.epoch, self.scheduler.get_lr()))
        # lr = self.scheduler.get_lr()[0]

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

    def log_training_results(self, engine, evaluator, train_loader):
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

    def log_validation_results(self, engine, evaluator, val_loader):
        pbar = self.log_cfg['pbar']
        # -----------------------------------------------
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_loss))

        pbar.n = pbar.last_print_n = 0
