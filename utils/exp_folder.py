from pathlib import Path
from datetime import datetime


def make_exp_folder(exp_folder):
    # Move it if folder exists
    folder = Path(exp_folder)
    if folder.exists():
        folder_suffix = datetime.utcnow().strftime("%y%m%d_%H%M%S")
        folder.rename(str(folder) + '_backup_at_' + folder_suffix)

    # mkdir -p <exp_folder>
    exp_folder = Path(exp_folder)
    exp_folder.mkdir(parents=True)
    return exp_folder
