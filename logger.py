import logging


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

def print_configs(logger, configs, task):
    logger.info("\n{} Configs: \n\t dataset_name - {}\n\t Method - {}\n\t Loss Function - {}\n\t Optimizer - {}\n\t init_lr - {}\n\t weight_decay - {}\n\t channel - {}\n\t batch_size - {}\n\t rotate - {}\n\t resize - {}\n\t centercrop - {}\n\t patch_size - {}\n\t epochs - {}\n\t Eval kernel size - {}\n".format(
        task,
        configs.dataset_name, 
        configs.model.__class__.__name__,
        configs.loss, 
        configs.optimizer,
        configs.init_lr, 
        configs.weight_decay, 
        configs.channel,
        configs.batch_size,
        configs.rotate, 
        configs.resize,
        configs.centercrop,
        configs.patch_size, 
        configs.epochs,
        configs.eval_kernel)
    )
