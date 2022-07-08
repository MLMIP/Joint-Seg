def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def adjust_lr(optimizer, base_lr, iter, max_iter, power):
    lr = base_lr * (1.0 - float(iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
