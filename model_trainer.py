import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import models
from models.__init__ import weight_init
from torchvision import transforms
from torch.utils.data import DataLoader
import utils

import os.path as osp
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from utils.logger import AverageMeter as meter
from data_loader import NUSIMG_Dataset, Office_Dataset, MRC_Dataset

from models.component import ProjectNetwork, Classifier


class ModelTrainer():
    def __init__(self, args, data, label_flag=None, v=None, logger=None):
        self.args = args
        self.batch_size = args.batch_size
        self.data_workers = 6

        self.data = data
        self.label_flag = label_flag

        self.num_class = data.num_class
        self.num_task = args.batch_size
        self.num_to_select = 0

        #GNN
        self.gnnModel = models.create('gnn', args).cuda()
        self.projector = ProjectNetwork(self.args, self.args.src_channel, self.args.tar_channel).cuda()
        self.classifier = Classifier(self.args).cuda()
        self.meter = meter(args.num_class)
        self.v = v

        self.criterionCE = nn.NLLLoss(reduction='mean').cuda()

        # BCE for edge
        self.criterion = nn.BCELoss(reduction='mean').cuda()
        self.global_step = 0
        self.logger = logger
        self.val_acc = 0
        # self.threshold = args.threshold


    def get_dataloader(self, dataset, training=False):
        data_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.data_workers,
                                 shuffle=training, pin_memory=True, drop_last=training)
        return data_loader

    def label2edge(self, targets):
        '''
        creat initial edge map and edge mask for unlabeled targets
        '''
        batch_size, num_sample = targets.size()
        # Target sample that notyet pseudo labeled
        target_node_mask = torch.eq(targets, self.num_class).type(torch.bool).cuda()
        source_node_mask = ~target_node_mask #& ~torch.eq(targets, self.num_class).type(torch.bool)

        label_i = targets.unsqueeze(-1).repeat(1, 1, num_sample)
        label_j = label_i.transpose(1, 2)

        edge = torch.eq(label_i, label_j).float().cuda()
        # unlabeled flag
        target_edge_mask = (torch.eq(label_i, self.num_class) + torch.eq(label_j, self.num_class)).type(torch.bool).cuda()
        source_edge_mask = ~target_edge_mask
        init_edge = (edge*source_edge_mask.float())
        return init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask

    def mmd_linear(self, src_fea, tar_fea):
        delta = (src_fea - tar_fea).squeeze(0)
        loss = torch.pow(torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1))),2)
        return torch.sqrt(loss)

    def transform_shape(self, tensor):

        batch_size, num_class, other_dim = tensor.shape
        tensor = tensor.view(1, batch_size * num_class, other_dim)
        return tensor

    def train(self, epochs=70, step=0):
        args = self.args
        self.step = step
        train_loader = self.get_dataloader(self.data, training=True)

        # initialize model

        # change the learning rate

        param_groups = [
                {'params': self.gnnModel.parameters()},
                {'params': self.projector.parameters()},
        ]


        if step > 0:
            lr = args.lr / (2 * step)
        else:
            lr = args.lr

        self.optimizer = torch.optim.Adam(params=param_groups,
                                          lr=lr,
                                          weight_decay=args.weight_decay)

        self.gnnModel.train()
        self.projector.train()
        self.meter.reset()

        for epoch in range(epochs):
            for i, inputs in enumerate(train_loader):

                src_fea = Variable(inputs[0][0]).cuda()
                tar_fea = Variable(inputs[0][1]).cuda()
                src_label = Variable(inputs[1][0]).cuda()
                tar_label = Variable(inputs[1][1]).cuda()
                num_task = src_fea.size(0)
                num_data = src_fea.size(1)

                # targets: real src label + pseudo target label
                targets = self.transform_shape(torch.cat([src_label, tar_label], axis=1).unsqueeze(-1)).squeeze(-1)


                init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = self.label2edge(targets)
                target_node_mask = inputs[4].view(-1) == 1
                known_label_mask = ((inputs[3] > 0).view_as(source_node_mask)).cuda()
                # temp_mask = known_label_mask[:, target_node_mask].squeeze(0)

                # Project Features into shared subspace
                src_fea, tar_fea, combine_fea = self.projector(src_fea, tar_fea)

                known_src_centroid, class_src = self.calculate_class_centriod(src_fea,
                                                                              targets[:, ~target_node_mask])
                kl = self.calculate_t_distribution(tar_fea, known_src_centroid)

                # feed into graph networks
                src_fea = src_fea.view(num_task, num_data, src_fea.size(1))
                tar_fea = tar_fea.view(num_task, num_data, tar_fea.size(1))
                edge_logits, node_logits = self.gnnModel(src_feat=src_fea, tar_feat=tar_fea,
                                                         init_edge_feat=init_edge, target_mask=target_node_mask)

                # compute edge loss
                full_edge_loss = [self.criterion(edge_logit.masked_select(source_edge_mask), init_edge.masked_select(source_edge_mask)) for edge_logit in edge_logits]

                # Linear MMD
                full_node_mmd = [self.mmd_linear(node_logit[:, ~target_node_mask],node_logit[:, target_node_mask]) for node_logit in node_logits]

                norm_node_logits = F.softmax(node_logits[-1], dim=-1)


                source_node_loss = self.criterionCE(torch.log(norm_node_logits[known_label_mask, :] + 1e-5),
                                                        targets.masked_select(known_label_mask))

                edge_loss = 0
                node_mmd = 0
                for l in range(args.num_layers - 1):
                    edge_loss += full_edge_loss[l]
                    node_mmd += full_node_mmd[l]
                edge_loss += full_edge_loss[-1] * 1
                loss =  args.edge_loss* edge_loss + args.node_loss * source_node_loss  + args.dis_loss * node_mmd + args.c_loss * kl

                node_pred = norm_node_logits[source_node_mask, :].detach().cpu().max(1)[1]
                node_prec = node_pred.eq(targets.masked_select(source_node_mask).detach().cpu()).double().mean()

                # Only for debugging
                if target_node_mask.any():
                    target_labels = Variable(inputs[2]).cuda()
                    target_labels = self.transform_shape(target_labels.unsqueeze(-1)).view(-1)
                    target_pred = norm_node_logits[:,target_node_mask, :].squeeze(0).max(1)[1]

                    # only predict on <labeled> data
                    target_prec = target_pred.eq(target_labels).double().data.cpu()

                    # update prec calculation on <labeled> data
                    self.meter.update(target_labels.detach().cpu().view(-1).numpy(),
                                      target_prec.numpy())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # if i > 150:
                #     break
            if (epoch + 1) % args.log_epoch == 0:
                print('---- Start Epoch {} Training --------'.format(epoch))
                print('Training Loss {:.3f}'.format(loss.data.cpu().numpy()))

                for k in range(args.num_class):
                    print('Class {} Precision: {:.3f}'.format(args.class_name[k], self.meter.avg[k]))
                self.meter.reset()

        # save model
        states = {
                  'projector': self.projector.state_dict(),
                  'graph': self.gnnModel.state_dict(),
                  'optimizer': self.optimizer.state_dict()}
        torch.save(states, osp.join(args.checkpoints_dir, '{}_step.pth.tar'.format(args.dataset)))
        self.meter.reset()

    """
        Return Classes N * Feature dim
    """
    def calculate_class_centriod(self, fea, label):
        class_fea = []
        class_label = []
        for i in range(self.num_class):
            temp_idx = torch.where(label == i)[1]
            if temp_idx.size(0) != 0:
                fea_sum = fea[temp_idx].mean(axis=0)
                class_fea.append(fea_sum)
                class_label.append(i)
        class_fea = torch.stack(class_fea)

        return class_fea, class_label
    """
        Return Classes N * Feature dim
    """
    def calculate_class_centriod(self, fea, label):
        class_fea = []
        class_label = []
        for i in range(self.num_class):
            temp_idx = torch.where(label == i)[1]
            if temp_idx.size(0) != 0:
                fea_sum = fea[temp_idx].mean(axis=0)
                class_fea.append(fea_sum)
                class_label.append(i)
        class_fea = torch.stack(class_fea)

        return class_fea, class_label


    def calculate_centroid_dist(self, src_cent, tar_label_fea, tar_label):
        class_dist = []
        for i in range(self.num_class):
            temp_idx = torch.where(tar_label == i)[1]
            if temp_idx.size(0) != 0:
                dist = F.pairwise_distance(src_cent[i], tar_label_fea[temp_idx]).mean()
                class_dist.append(dist)
        if len(class_dist) > 0:
            return torch.stack(class_dist).sum()
        else:
            return 0

    def calculate_qk(self, tar_fea, src_cent, class_idx):
        top = 1/ (1 + F.pairwise_distance(tar_fea, src_cent[class_idx]))
        bottom = []
        for i in range(self.num_class):
            bottom.append(1/(1 + F.pairwise_distance(tar_fea, src_cent[i])))
        bottom = torch.stack(bottom)
        return top / bottom.sum(axis=0)

    def calculate_pk(self,q, class_idx):
        top = q[class_idx]**2 / q[class_idx].sum(axis=0)
        bottom = []
        for i in range(self.num_class):
            bottom.append(q[i]**2 / q[i].sum(axis=0))
        bottom = torch.stack(bottom)
        return top / bottom.sum(axis=0)


    def calculate_t_distribution(self, tar_fea, src_cent):
        q = []
        kl = []
        for k in range(self.num_class):
            q.append(self.calculate_qk(tar_fea, src_cent, k))
        for k in range(self.num_class):
            pk = self.calculate_pk(q, k)
            kl.append(pk * torch.log(pk/q[k]))
        kl = torch.stack(kl)
        return kl.sum()


    def estimate_label(self):

        args = self.args
        print('Evaluation & Pseudo Labeling...')
        if args.dataset == 'nusimg':
            test_data = NUSIMG_Dataset(root=args.data_dir, partition='test', label_flag=self.label_flag,
                                       source=args.source_path, target=args.target_path)
        elif args.dataset == 'office':
            test_data = Office_Dataset(root=args.data_dir, partition='test', label_flag=self.label_flag,
                                       source=args.source_path, target=args.target_path)
        elif args.dataset == 'mrc':

            test_data = MRC_Dataset(root=args.data_dir, partition='test', label_flag=self.label_flag,
                                       source=args.source_path, target=args.target_path)

        self.meter.reset()
        # append labels and scores for target samples
        pred_labels = []
        pred_scores = []
        real_labels = []
        target_loader = self.get_dataloader(test_data, training=False)
        self.gnnModel.eval()
        self.projector.eval()

        num_correct = 0

        for i, inputs in enumerate(target_loader):

            src_fea = Variable(inputs[0][0], requires_grad=False).cuda()
            tar_fea = Variable(inputs[0][1], requires_grad=False).cuda()
            src_label = Variable(inputs[1][0]).cuda()
            tar_label = Variable(inputs[1][1]).cuda()
            real_tar_label = Variable(inputs[2]).cuda()
            num_task = src_fea.size(0)
            num_data = src_fea.size(1)

            # only for debugging
            targets = self.transform_shape(torch.cat([src_label, tar_label], axis=1).unsqueeze(-1)).squeeze(-1)
            init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = self.label2edge(targets)
            target_node_mask = inputs[4].view(-1) == 1
            # feed into project networks
            src_fea, tar_fea, _ = self.projector(src_fea, tar_fea)
            # feed into graph networks
            src_fea = src_fea.view(num_task, num_data, src_fea.size(1))
            tar_fea = tar_fea.view(num_task, num_data, tar_fea.size(1))
            edge_logits, node_logits = self.gnnModel(src_feat=src_fea, tar_feat=tar_fea,
                                                     init_edge_feat=init_edge, target_mask=target_node_mask)
            norm_node_logits = F.softmax(node_logits[-1], dim=-1)
            target_score, target_pred = norm_node_logits[:,target_node_mask, :].squeeze(0).max(1)

            pred = target_pred.detach().cpu()
            target_prec = pred.eq(real_tar_label.view(-1).detach().cpu()).double()

            self.meter.update(
                real_tar_label.detach().cpu().view(-1).data.cpu().numpy(),
                target_prec.numpy())



            pred_labels.append(target_pred.cpu().detach())
            pred_scores.append(target_score.cpu().detach())
            real_labels.append(real_tar_label.cpu().detach())

        pred_labels = torch.cat(pred_labels)
        pred_scores = torch.cat(pred_scores)
        real_labels = torch.cat(real_labels)



        self.gnnModel.train()
        self.projector.train()
        # TO-DO
        self.num_to_select = int(len(target_loader) * self.args.batch_size * self.args.num_class * self.args.EF / 100)
        return pred_labels.data.cpu().numpy(), pred_scores.data.cpu().numpy(), real_labels.data.cpu().numpy()

    def select_top_data(self, pred_score):
        # remark samples if needs pseudo labels based on classification confidence
        if self.v is None:
            self.v = np.zeros(len(pred_score))
        unselected_idx = np.where(self.v == 0)[0]
        if len(unselected_idx) < self.num_to_select:
            self.num_to_select = len(unselected_idx)
        index = np.argsort(-pred_score[unselected_idx])
        index_orig = unselected_idx[index]
        for i in range(self.num_to_select):
            self.v[index_orig[i]] = 1
        return self.v

    def generate_new_train_data(self, sel_idx, pred_y, real_label):
        # create the new dataset merged with pseudo labels
        assert len(sel_idx) == len(pred_y)
        new_label_flag = []
        pos_correct, pos_total, neg_correct, neg_total = 0, 0, 0, 0
        real_label = np.reshape(real_label, -1)
        for i, flag in enumerate(sel_idx):
            if i >= len(real_label):
                break
            if flag > 0:
                new_label_flag.append(pred_y[i])
                pos_total += 1
                if real_label[i] == pred_y[i]:
                    pos_correct += 1
            else:
                new_label_flag.append(self.args.num_class)


        self.meter.reset()
        self.meter.update(real_label, (pred_y == real_label).astype(int))

        for k in range(self.args.num_class):
            print('Target {} Precision: {:.3f}'.format(self.args.class_name[k], self.meter.avg[k]))

        print('Class Averaged accuracy = {:0.4f}, Accuracy = {:0.4f}'.format(self.meter.avg.mean(),
                                                                                       self.meter.sum.sum() / self.meter.count.sum()))

        correct = pos_correct + neg_correct
        total = pos_total + neg_total
        acc = correct / total
        new_label_flag = torch.tensor(new_label_flag)

        # update source data
        if self.args.dataset == 'nusimg':
            new_data = NUSIMG_Dataset(root=self.args.data_dir, partition='train', label_flag=new_label_flag,
                                      source=self.args.source_path, target=self.args.target_path, target_ratio=self.step)

        elif self.args.dataset == 'office':
            new_data = Office_Dataset(root=self.args.data_dir, partition='train', label_flag=new_label_flag,
                                       source=self.args.source_path, target=self.args.target_path, target_ratio=self.step)

        elif self.args.dataset == 'mrc':
            new_data = MRC_Dataset(root=self.args.data_dir, partition='train', label_flag=new_label_flag,
                                       source=self.args.source_path, target=self.args.target_path, target_ratio=self.step)

        print('selected pseudo-labeled data: {} of {} is correct, accuracy: {:0.4f}'.format(correct, total, acc))
        self.label_flag = new_label_flag
        self.data = new_data
        return new_label_flag

    def one_hot_encode(self, num_classes, class_idx):
        return torch.eye(num_classes, dtype=torch.long)[class_idx]

    def load_model_weight(self, path):
        print('loading weight')
        state = torch.load(path)
        self.projector.load_state_dict(state['projector'])
        self.gnnModel.load_state_dict(state['graph'])

    def label2edge_gt(self, targets):
        '''
        creat initial edge map and edge mask for unlabeled targets
        '''
        batch_size, num_sample = targets.size()
        target_node_mask = torch.eq(targets, self.num_class).type(torch.bool).cuda()
        source_node_mask = ~target_node_mask & ~torch.eq(targets, self.num_class - 1).type(torch.bool)

        label_i = targets.unsqueeze(-1).repeat(1, 1, num_sample)
        label_j = label_i.transpose(1, 2)

        edge = torch.eq(label_i, label_j).float().cuda()
        target_edge_mask = (torch.eq(label_i, self.num_class) + torch.eq(label_j, self.num_class)).type(
            torch.bool).cuda()
        source_edge_mask = ~target_edge_mask
        # unlabeled flag

        return (edge*source_edge_mask.float())

    def extract_feature(self):
        print('Feature extracting...')
        import scipy.io as sio
        self.meter.reset()
        # append labels and scores for target samples
        src_fea_list = []
        target_fea_list = []
        src_labels = []
        tar_labels = []
        overall_split = []
        target_loader = self.get_dataloader(self.data, training=False)
        self.projector.eval()
        self.gnnModel.eval()
        num_correct = 0
        skip_flag = self.args.visualization
        with tqdm(total=len(target_loader)) as pbar:
            for i, inputs in enumerate(target_loader):

                src_fea = Variable(inputs[0][0], requires_grad=False).cuda()
                tar_fea = Variable(inputs[0][1], requires_grad=False).cuda()
                src_label = Variable(inputs[1][0]).cuda()
                tar_label = Variable(inputs[1][1]).cuda()

                num_task = src_fea.size(0)
                num_data = src_fea.size(1)
                targets = self.transform_shape(torch.cat([src_label, tar_label], axis=1).unsqueeze(-1)).squeeze(-1)
                init_edge, target_edge_mask, source_edge_mask, target_node_mask, source_node_mask = self.label2edge(
                    targets)
                target_node_mask = inputs[4].view(-1) == 1
                # feed into project networks
                src_fea, tar_fea, _ = self.projector(src_fea, tar_fea)
                # feed into graph networks
                src_fea = src_fea.view(num_task, num_data, src_fea.size(1))
                tar_fea = tar_fea.view(num_task, num_data, tar_fea.size(1))
                edge_logits, node_logits = self.gnnModel(src_feat=src_fea, tar_feat=tar_fea,
                                                         init_edge_feat=init_edge, target_mask=target_node_mask)



                src_labels.append(src_label.view(1, num_task*num_data).data.cpu())
                tar_labels.append(tar_label.view(1, num_task*num_data).data.cpu())
                src_fea_list.append(node_logits[-1][:, ~target_node_mask].squeeze(0).data.cpu())
                target_fea_list.append(node_logits[-1][:,target_node_mask].squeeze(0).data.cpu())
                pbar.update()

        src_labels = torch.cat(src_labels, axis=1).numpy()
        tar_labels = torch.cat(tar_labels, axis=1).numpy()
        src_fea_list = torch.cat(src_fea_list).numpy()
        target_fea_list = torch.cat(target_fea_list).numpy()
        mat_dict = {'source_label': src_labels, 'source_fea':src_fea_list, 'target_label':tar_labels , 'target_fea':target_fea_list}
        sio.savemat('learned_fea', mat_dict)
        return mat_dict









