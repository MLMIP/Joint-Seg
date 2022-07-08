import os
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import time
import shutil
import datetime
from absl import logging
import torch
import warnings
from torch import optim
from torchsummary import summary

from dataset import load_data
from train import train
from test import test_train, test_test
from evaluate import evaluate
from visualizer import Visualizer
from logger import get_logger, print_configs
from configs import configs


def main(configs, viz, logger, model, trainSet, validSet):

    if configs.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=configs.init_lr, weight_decay=configs.weight_decay)
    elif configs.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=configs.init_lr, momentum=0.99, weight_decay=configs.weight_decay)
    
    best = {"epoch_faz": 0, "dice_faz": 0, "epoch_rv": 0, "dice_rv": 0}
    
    logger.info("-------Training Started...-------\n")
    start = time.time()
    for epoch in range(configs.epochs):
        logger.info("[Epoch {}/{} on Network-{} and DataSet-{}]".format(
            epoch+1, configs.epochs, configs.model.__class__.__name__, configs.dataset_name)
        )
        model = train(configs, viz, logger, trainSet, model, optimizer, epoch+1)
        if (epoch+1) % 1 ==  0 or epoch == configs.epochs-1:
            model, best = evaluate(configs, viz, logger, best, validSet, model, epoch+1, isSave=False)
        logger.info("\n")
    end = time.time()
    logger.info("-------Training Completed!!! Time Cost: %d s-------\n" % (end-start))


warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

now = datetime.datetime.now()
log_file_path = './exp_logs/' + '%s-%04d%02d%02d-%02d%02d.log' % (configs.dataset_name[-2:], now.year, now.month, now.day, now.hour, now.minute)
logger = get_logger(log_file_path)
print_configs(logger, configs, 'Joint Segmentation')

viz = Visualizer(env=configs.data_dir.split('/')[-1], port=9999)

trainSet, validSet, testSet = load_data(configs)

if configs.mode == 'TRAIN': # train & test best model
    # train
    model = configs.model
    main(configs, viz, logger, model, trainSet, validSet)

    # test
    model_faz = torch.load(configs.models_save_dir + '/model-best-faz.pth')
    model_rv = torch.load(configs.models_save_dir + '/model-best-rv.pth')
    logger.info("-------Evaluating on best model...-------")

    if os.path.exists(configs.results_save_dir + "/out"):
        shutil.rmtree(configs.results_save_dir + "/out")

    dice_dct_faz, dice_dct_rv = test_test(configs, viz, logger, testSet, model_faz, model_rv, isSave=True)
    test_dice_faz = round(dice_dct_faz["out"].mean() + 1e-12, 4)
    test_dice_rv = round(dice_dct_rv["out"].mean() + 1e-12, 4)

    os.rename(log_file_path, log_file_path[:-4] + '-{}-{}-{}.log'.format(
        str(test_dice_faz), str(test_dice_rv), configs.model.__class__.__name__
    ))
    os.rename(configs.models_save_dir + '/model-best-faz.pth', configs.models_save_dir + '/model-{}-faz.pth'.format(str(test_dice_faz)))
    os.rename(configs.models_save_dir + '/model-best-rv.pth', configs.models_save_dir + '/model-{}-rv.pth'.format(str(test_dice_rv)))
