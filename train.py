import torch
import torch.nn as nn
from tqdm import tqdm
from loss import dice_loss
from utils import get_lr, adjust_lr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def train(configs, viz, logger, trainSet, model, optimizer, epoch):

    if configs.model.__class__.__name__ in ['base', 'full']:
        model = train(configs, viz, logger, trainSet, model, optimizer, epoch)
    
    else:
        raise NotImplementedError("train process for {} is not implemented.".format(configs.model.__class__.__name__))
    
    return model


def train_common(configs, viz, logger, trainSet, model, optimizer, epoch):
    print("Training...")
    BCE_Loss = nn.BCELoss()
    Dice_Loss = dice_loss()
    
    epoch_loss = 0

    for sample in tqdm(trainSet):
        img = sample[0].to(device)
        faz_gt = sample[1].to(device)
        rv_gt = sample[2].to(device)
        optimizer.zero_grad()

        faz_pred, rv_pred = model(img)
        
        # Visualizer 
        viz.img(name="train_images", img_=img)
        viz.img(name="train_labels_faz", img_=faz_gt)
        viz.img(name="train_labels_rv", img_=rv_gt)
        viz.img(name="train_prediction_faz", img_=faz_pred)
        viz.img(name="train_prediction_rv", img_=rv_pred)
        
        faz_loss = BCE_Loss(faz_pred, faz_gt) + Dice_Loss(faz_pred, faz_gt)
        rv_loss = BCE_Loss(rv_pred, rv_gt)

        weighted_loss = 0.5 * faz_loss +  + 0.5 * rv_loss
        
        weighted_loss.backward(retain_graph=True)
        optimizer.step()
        epoch_loss += weighted_loss.item()
        
        current_lr = get_lr(optimizer)
       
    viz.plot("epoch loss", epoch_loss/len(trainSet))
    logger.info("[Epoch %d/%d Loss: %0.4f, Learning Rate: %f]" % (epoch, configs.epochs, epoch_loss/len(trainSet), current_lr))
    adjust_lr(optimizer, configs.init_lr, epoch, configs.epochs, configs.power)
    
    return model
