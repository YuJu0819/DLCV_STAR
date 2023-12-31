import torch
import math
import sys
from typing import Iterable
import util.misc as misc
import util.lr_sched as lr_sched
from tqdm import tqdm


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, loss_scaler, args=None):
    model.train(True)
    # metric_logger = misc.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # header = 'Epoch: [{}]'.format(epoch)
    # print_freq = int(len(data_loader)/2)
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    # for data_iter_step, data in enumerate(metric_logger.log_every(tqdm(data_loader), print_freq, header)):
    for data_iter_step, data in enumerate(tqdm(data_loader)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        vqa_loss, vaq_loss, qav_loss = model(data)

        loss = vqa_loss + vaq_loss + qav_loss
        loss_value = loss.item()
        # vqa_loss_value = vqa_loss.item()
        # vaq_loss_value = vaq_loss.item()
        # qav_loss_value = qav_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # torch.cuda.synchronize()

        # metric_logger.update(loss=loss_value)
        # metric_logger.update(vqa_loss=vqa_loss_value)
        # metric_logger.update(vaq_loss=vaq_loss_value)
        # metric_logger.update(qav_loss=qav_loss_value)
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return loss_value


def val_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, args=None):
    model.eval()
    # metric_logger = misc.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # header = 'Epoch: [{}]'.format(epoch)
    # print_freq = int(len(data_loader)/8)

    # for data_iter_step, data in enumerate(metric_logger.log_every(tqdm(data_loader), print_freq, header)):
    accs = 0
    num=0
    for data_iter_step, data in enumerate(tqdm(data_loader)):
        answer = data['answer'].cuda()
        bsz = answer.shape[0]

        with torch.no_grad():
            logits = model(data, inference=True)
        
        count = (logits != 0).sum(-1)
        prediction = (logits.sum(-1) / count).argmin(-1)
        #print("answer",answer)
        #print("prediction",prediction)
        
        eval = (answer == prediction)
        acc = eval.sum().item() / bsz
        accs=accs+acc
        num+=1
        # misc.log_qtype(data, eval, metric_logger, args)

        # lr = optimizer.param_groups[0]["lr"]
        # metric_logger.update(lr=lr)
        # metric_logger.update(n=bsz, acc=acc)
    accs = accs/num
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return accs
