import logging
import os


def makedirs(path):
    if path != '':
        os.makedirs(path, exist_ok=True)


def create_logger(name, save_dir=None):
    logger = logging.getLogger(name)

    if logging.getLogger().hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)
        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.INFO)
        logger.addHandler(fh_v)

    return logger

