import os
import pathlib
import sys

from src.estimater import Model, Processor
from src.preprocess import PrepareData
from src.preprocess import Processor as Preprocessor
from src.util import logger, tmp_file


def prepare_data():
    tmp_file.create_workdir()
    PrepareData().dl()


def preprocess():
    Preprocessor(tmp_file.raw_trainset()).execute(
        ).save_dataset(tmp_file.trainset())
    Preprocessor(tmp_file.raw_estimate_set()).execute(
        train=False).save_dataset(tmp_file.estimate_set())


def train():
    Processor().train()


def cross_validation():
    Processor().cross_validation()


def predict():
    Processor(Model.load()).predict()

if __name__ == '__main__':
    logger = logger.get_logger('main')
    args = sys.argv
    try:
        task = args[1]
    except IndexError:
        logger.info('Please set task in args.')
        sys.exit()
    if task == 'prepare_data':
        logger.info('Execute task: prepare data.')
        prepare_data()
    elif task == 'preprocess':
        logger.info('Execute task: preprocess.')
        preprocess()
    elif task == 'cross_validation':
        logger.info('Execute task: cross_validation.')
        cross_validation()
    elif task == 'train':
        logger.info('Execute task: train.')
        train()
    elif task == 'predict':
        logger.info('Execute task: predict.')
        predict()
    else:
        logger.info('The task is not defined.')
else:
    import digdag
    current_dir = pathlib.Path(__file__).resolve().parent
    sys.path.append(str(current_dir))
