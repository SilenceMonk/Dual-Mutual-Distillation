import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from networks.vnet_sdf import VNet
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
import monai

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/Dual-Mutual-Distillation-master/data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='DMD', help='model_name')
parser.add_argument('--max_iterations', type=int, default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=8, help='labelled number of training examples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='2', help='GPU to use')

### costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--kd_type', type=str, choices=['dice', 'ce', 'kl'], default="dice",
                    help='loss type for loss_kd')
parser.add_argument('--lam_kd', type=float, default=4, help='trade-off weight')
parser.add_argument('--T', type=float, default=2, help='temperature')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if not args.deterministic:
    cudnn.benchmark = True  #
    cudnn.deterministic = False  #
else:
    cudnn.benchmark = False  # True #
    cudnn.deterministic = True  # False #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    def create_model(ema=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes - 1, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


    model_1 = create_model()
    model_2 = create_model()

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',  # train/val split
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))

    labelnum = args.labelnum  # default 16
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, 80))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    param_list = [{'params': model_1.parameters()}, {'params': model_2.parameters()}]
    optimizer = optim.SGD(param_list, lr=base_lr, momentum=0.9, weight_decay=0.0001)

    model_1.train()
    model_2.train()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs_1 = model_1(volume_batch)
            outputs_soft_1 = torch.sigmoid(outputs_1)
            outputs_2 = model_2(volume_batch)
            outputs_soft_2 = torch.sigmoid(outputs_2)

            outputs_soft_1_dis = torch.sigmoid(outputs_1 / args.T)
            outputs_soft_2_dis = torch.sigmoid(outputs_2 / args.T)

            loss_seg_dice = losses.dice_loss(outputs_soft_1[:labeled_bs], label_batch[:labeled_bs]) + \
                            losses.dice_loss(outputs_soft_2[:labeled_bs], label_batch[:labeled_bs])

            supervised_loss = loss_seg_dice

            # knowledge distillation loss type
            if args.kd_type == 'kl':
                loss_kd = F.kl_div(F.logsigmoid(outputs_1 / args.T), outputs_soft_2_dis, reduction='mean') + \
                          F.kl_div(F.logsigmoid(outputs_2 / args.T), outputs_soft_1_dis, reduction='mean')

            elif args.kd_type == 'ce':
                loss_kd = (F.logsigmoid(outputs_1 / args.T) * outputs_soft_2_dis +
                           F.logsigmoid(outputs_2 / args.T) * outputs_soft_1_dis).mean()

            elif args.kd_type == 'dice':
                loss_kd = losses.dice_loss(outputs_soft_1_dis, outputs_soft_2_dis)
            else:
                loss_kd = None

            outputs_1.requires_grad_(True)
            outputs_1.retain_grad()

            loss = supervised_loss + args.lam_kd * loss_kd
            optimizer.zero_grad()                    
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss_kd', loss_kd, iter_num)
  

            logging.info(
                'iteration %d : loss : %f, loss_dice: %f, loss_kd: %f' %
                (iter_num, loss.item(), loss_seg_dice.item(), loss_kd.item()))

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                # save_mode_path = os.path.join(snapshot_path, 'latest.pth')
                torch.save(model_1.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
