import os
import torch
from test import test_train

def evaluate(configs, viz, logger, best, validSet, model, epoch, isSave):
    model.eval()
    
    _, _, _, _, _, dice_dct_faz, _, _, _, _, _, _, dice_dct_rv, _ = test_train(configs, viz, logger, validSet, model, isSave)
    
    # auc = round(auc_dct["out"].mean() + 1e-12, 4)
    dice_faz = round(dice_dct_faz["out"].mean() + 1e-12, 4)
    dice_rv = round(dice_dct_rv["out"].mean() + 1e-12, 4)
    
    # os.mkdir(configs.models_save_dir)
    if dice_faz >= best["dice_faz"]:
        best["epoch_faz"] = epoch
        best["dice_faz"] = dice_faz
        torch.save(model, os.path.join(configs.models_save_dir, "model-best-faz.pth"))
        logger.info("Best FAZ Model Updated!")

    if dice_rv >= best["dice_rv"]:
        best["epoch_rv"] = epoch
        best["dice_rv"] = dice_rv
        torch.save(model, os.path.join(configs.models_save_dir, "model-best-rv.pth"))
        logger.info("Best RV Model Updated!")
    
    logger.info("[Best: FAZ - Epoch %d/%d - Dice: %.4f, RV - Epoch %d/%d - Dice: %.4f]" % (best["epoch_faz"], configs.epochs, best["dice_faz"], best["epoch_rv"], configs.epochs, best["dice_rv"]))
    
    if epoch == configs.epochs:
        torch.save(model, os.path.join(configs.models_save_dir, "model-latest-faz.pth"))
        torch.save(model, os.path.join(configs.models_save_dir, "model-latest-rv.pth"))
        
    model.train(mode=True)
    model.train(mode=True)
    
    return model, best
