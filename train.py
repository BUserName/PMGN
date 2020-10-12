from __future__ import print_function, absolute_import
import os
import argparse
import random
import numpy as np
import torch
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# data
from data_loader import Office_Dataset, NUSIMG_Dataset, MRC_Dataset
from model_trainer import ModelTrainer

# Modified here
from utils.logger import Logger

def main(args):
    # Modified here
    total_step = 100//args.EF

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # prepare checkpoints and log folders
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)

    # initialize dataset
    if args.dataset == 'nusimg':
        args.data_dir = os.path.join(args.data_dir, 'visda')
        data = NUSIMG_Dataset(root=args.data_dir, partition='train', label_flag=None, source=args.source_path,
                              target=args.target_path)

    elif args.dataset == 'office':
        args.data_dir = os.path.join(args.data_dir, 'Office')
        data = Office_Dataset(root=args.data_dir, partition='train', label_flag=None, source=args.source_path,
                              target=args.target_path)
    elif args.dataset == 'mrc':
        data = MRC_Dataset(root=args.data_dir, partition='train', label_flag=None, source=args.source_path,
                              target=args.target_path)
    else:
        print('Unknown dataset!')

    args.class_name = data.class_name
    args.num_class = data.num_class
    args.alpha = data.alpha
    # setting experiment name
    label_flag = None
    selected_idx = None
    args.experiment = set_exp_name(args)

    logger = Logger(args)
    trainer = ModelTrainer(args=args, data=data, label_flag=label_flag, v=selected_idx, logger=logger)
    for step in range(total_step):

        print("This is {}-th step with EF={}%".format(step, args.EF))
        # train the model
        args.log_epoch = 5
        trainer.train(epochs= 24,step=step)#24
        # psedo_label
        pred_y, pred_score, pred_acc = trainer.estimate_label()

        # select data from target to source
        selected_idx = trainer.select_top_data(pred_score)

        # add new data
        trainer.generate_new_train_data(selected_idx, pred_y, pred_acc)


def set_exp_name(args):
    exp_name = 'D-{}'.format(args.dataset)
    if args.dataset == 'office' or args.dataset == 'home':
        exp_name += '_src-{}_tar-{}'.format(args.source_path, args.target_path)
    exp_name += '_A-{}'.format(args.arch)
    exp_name += '_L-{}'.format(args.num_layers)
    exp_name += '_E-{}_B-{}'.format(args.EF, args.batch_size)
    return exp_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exploit Domain Adaptation')
    # set up dataset & backbone embedding
    dataset = 'office'
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('-a', '--arch', type=str, default='res')
    parser.add_argument('--root_path', type=str, default='./utils/', metavar='B',
                        help='root dir')

    # set up path
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'data/'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'logs'))
    parser.add_argument('--checkpoints_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'checkpoints'))

    # verbose setting
    parser.add_argument('--log_step', type=int, default=30)
    parser.add_argument('--log_epoch', type=int, default=3)

    if dataset == 'office':
        parser.add_argument('--source_path', type=str, default='/home/zijian/Desktop/HeteroDA/Hetero_DA/data/Office-Caltech/AMAZON_SURF.mat')
        parser.add_argument('--target_path', type=str, default='/home/zijian/Desktop/HeteroDA/Hetero_DA/data/Office-Caltech/WEBCAM_DECAF.mat')
        parser.add_argument('--src_channel', type=int, default='800')
        parser.add_argument('--tar_channel', type=int, default='4096')
        parser.add_argument('--edgeUpdate', type=int, default='1')
    elif dataset == 'nusimg':
        parser.add_argument('--source_path', type=str, default='/home/zijian/Desktop/HeteroDA/Hetero_DA/data/Office-Caltech/NUSTAG_HIST.mat')
        parser.add_argument('--target_path', type=str, default='/home/zijian/Desktop/HeteroDA/Hetero_DA/data/Office-Caltech/IMGNET_DECAF.mat')
        parser.add_argument('--src_channel', type=int, default='64')
        parser.add_argument('--tar_channel', type=int, default='4096')
        parser.add_argument('--edgeUpdate', type=int, default='0')
    elif dataset == 'mrc':
        parser.add_argument('--source_path', type=str,
                            default='/home/zijian/Desktop/HeteroDA/Hetero_DA/data/mrc/rand_GR.mat')
        parser.add_argument('--target_path', type=str,
                            default='/home/zijian/Desktop/HeteroDA/Hetero_DA/data/mrc/rand_SP_10.mat')
        parser.add_argument('--src_channel', type=int, default='1417') # E 1131, F 1230, G 1417, I 1041
        parser.add_argument('--tar_channel', type=int, default='807')
        parser.add_argument('--edgeUpdate', type=int, default='0')
    else:
        print("Unknown Dataset")
    parser.add_argument('--eval_log_step', type=int, default=100)
    parser.add_argument('--test_interval', type=int, default=1500)


    # hyper-parameters

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--EF', type=int, default=10)
    parser.add_argument('--loss', type=str, default='nll')

    # optimizer
    parser.add_argument('--lr', type=float, default=5e-4)#5e-4
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parser.add_argument('--in_features', type=int, default=512)

    parser.add_argument('--node_features', type=int, default=512)
    parser.add_argument('--edge_features', type=int, default=512)

    #GNN hyper-parameters
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--node_loss', type=float, default=1)
    parser.add_argument('--edge_loss', type=float, default=1.5)
    parser.add_argument('--dis_loss', type=float, default=0.25)
    parser.add_argument('--c_loss', type=float, default=100)
    main(parser.parse_args())
