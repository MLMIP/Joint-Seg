import torch
import os
import ml_collections
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

configs = ml_collections.ConfigDict()

configs.data_dir = './Data/OCTA-500/OCTA_6M'
# configs.data_dir = './Data/OCTA-500/OCTA_3M'

configs.DataParallel = False
configs.ContinueTraining = False
configs.mode = 'TEST'   # TRAIN or TEST

#----------------------Select Network-----------------------#
configs.model = Joint_Seg(1).to(device)
#----------------------Select Network-----------------------#

# ---------------------Hyper Parameters---------------------#
configs.dataset_name = configs.data_dir.split('/')[-1]
configs.eval_kernel = (1, 1)
configs.channel = 3
configs.batch_size = 2
configs.patch_size = None
configs.rotate = None
configs.resize = None 
configs.centercrop = None
configs.init_lr = 1e-4
configs.weight_decay = 1e-4
configs.power = 0.9
configs.epochs = 100
configs.threshold = 0.5
configs.optimizer = 'Adam'
configs.loss = 'bce_dice'         # select from dice / bce / mse / bce_dice
# ---------------------Hyper Parameters---------------------#

configs.models_save_dir = './models' + '/{}/{}'.format(
    configs.dataset_name, configs.model.__class__.__name__
)
if not os.path.exists(configs.models_save_dir):
    os.makedirs(configs.models_save_dir)

configs.results_save_dir = './results' + '/{}/{}'.format(
    configs.dataset_name, configs.model.__class__.__name__
)
if not os.path.exists(configs.results_save_dir):
    os.makedirs(configs.results_save_dir)
