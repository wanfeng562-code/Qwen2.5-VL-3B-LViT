# -*- coding: utf-8 -*-
import torch.optim
import os
import time
import numpy as np
from utils import *
import Config as config
import warnings
from torchinfo import summary
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
warnings.filterwarnings("ignore")


def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, hd, average_hd, acc, average_acc, mode, lr, logger):
    '''
        mode = Train or Test
    '''
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += 'Loss:{:.3f} '.format(loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += 'IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    string += 'HD:{:.3f} '.format(hd)
    string += '(Avg {:.4f}) '.format(average_hd)
    # string += 'Acc:{:.3f} '.format(acc)
    # string += '(Avg {:.4f}) '.format(average_acc)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)
    # print summary


def _hausdorff_distance(pred_binary, gt_binary, max_points=2000):
    pred_pts = np.argwhere(pred_binary > 0)
    gt_pts = np.argwhere(gt_binary > 0)

    if pred_pts.shape[0] == 0 and gt_pts.shape[0] == 0:
        return 0.0
    if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        h, w = pred_binary.shape
        return float(np.hypot(h, w))

    if pred_pts.shape[0] > max_points:
        idx = np.random.choice(pred_pts.shape[0], max_points, replace=False)
        pred_pts = pred_pts[idx]
    if gt_pts.shape[0] > max_points:
        idx = np.random.choice(gt_pts.shape[0], max_points, replace=False)
        gt_pts = gt_pts[idx]

    dist = cdist(pred_pts.astype(np.float32), gt_pts.astype(np.float32), metric='euclidean')
    return float(max(dist.min(axis=1).max(), dist.min(axis=0).max()))


def hd_on_batch(masks, pred):
    hds = []
    if pred.shape[1] > 1:
        pred_cls = torch.argmax(pred, dim=1)
        gt_cls = masks[:, 0].long() if (masks.dim() == 4 and masks.shape[1] == 1) else masks.long()
        num_classes = pred.shape[1]
        for i in range(pred.shape[0]):
            cls_hds = []
            for c in range(1, num_classes):
                pred_tmp = (pred_cls[i] == c).cpu().detach().numpy().astype(np.uint8)
                mask_tmp = (gt_cls[i] == c).cpu().detach().numpy().astype(np.uint8)
                cls_hds.append(_hausdorff_distance(pred_tmp, mask_tmp))
            hds.append(float(np.mean(cls_hds)) if len(cls_hds) > 0 else 0.0)
    else:
        for i in range(pred.shape[0]):
            pred_tmp = pred[i][0].cpu().detach().numpy()
            mask_tmp = masks[i].cpu().detach().numpy()
            pred_tmp = (pred_tmp >= 0.5).astype(np.uint8)
            mask_tmp = (mask_tmp > 0).astype(np.uint8)
            hds.append(_hausdorff_distance(pred_tmp, mask_tmp))
    return float(np.mean(hds)) if len(hds) > 0 else 0.0


##################################################################################
#=================================================================================
#          Train One Epoch
#=================================================================================
##################################################################################
def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger):
    logging_mode = 'Train' if model.training else 'Val'
    end = time.time()
    time_sum, loss_sum = 0, 0
    dice_sum, iou_sum, hd_sum, acc_sum = 0.0, 0.0, 0.0, 0.0
    dices = []
    for i, (sampled_batch, names) in enumerate(loader, 1):

        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # Take variable and put them to GPU
        images, masks, text = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
        if text.shape[1] > 10:
            text = text[ :, :10, :]
        
        images, masks, text = images.cuda(), masks.cuda(), text.cuda()


        # ====================================================
        #             Compute loss
        # ====================================================

        preds = model(images, text)
        out_loss = criterion(preds, masks.float())  # Loss
        # print(model.training)


        if model.training:
            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()

        train_dice = criterion._show_dice(preds, masks.float())
        train_iou = iou_on_batch(masks,preds)
        train_hd = hd_on_batch(masks, preds)

        batch_time = time.time() - end
        if epoch % config.vis_frequency == 0 and logging_mode is 'Val':
            vis_path = config.visualize_path+str(epoch)+'/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images,masks,preds,names,vis_path)
        dices.append(train_dice)

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        iou_sum += len(images) * train_iou
        hd_sum += len(images) * train_hd
        # acc_sum += len(images) * train_acc
        dice_sum += len(images) * train_dice

        if i == len(loader):
            average_loss = loss_sum / (config.batch_size*(i-1) + len(images))
            average_time = time_sum / (config.batch_size*(i-1) + len(images))
            train_iou_average = iou_sum / (config.batch_size*(i-1) + len(images))
            train_hd_average = hd_sum / (config.batch_size*(i-1) + len(images))
            # train_acc_average = acc_sum / (config.batch_size*(i-1) + len(images))
            train_dice_avg = dice_sum / (config.batch_size*(i-1) + len(images))
        else:
            average_loss = loss_sum / (i * config.batch_size)
            average_time = time_sum / (i * config.batch_size)
            train_iou_average = iou_sum / (i * config.batch_size)
            train_hd_average = hd_sum / (i * config.batch_size)
            # train_acc_average = acc_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / (i * config.batch_size)

        end = time.time()
        torch.cuda.empty_cache()

        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), out_loss, loss_name, batch_time,
                          average_loss, average_time, train_iou, train_iou_average,
                          train_dice, train_dice_avg, train_hd, train_hd_average, 0, 0,  logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups),logger=logger)

        if config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)

            # plot metrics in tensorboard
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            # writer.add_scalar(logging_mode + '_acc', train_acc, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)

        torch.cuda.empty_cache()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return average_loss, train_dice_avg, train_iou_average, train_hd_average, average_time
