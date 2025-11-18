# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from semilearn.algorithms.softmatch.utils import SoftMatchWeightingHook
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS, get_data_loader
from semilearn.algorithms.hooks import PseudoLabelingHook, DistAlignEMAHook
from semilearn.algorithms.utils import SSL_Argument, str2bool

import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, top_k_accuracy_score, roc_curve, auc, classification_report
from semilearn.core.criterions import CrossEntropySurvLoss, CoxPHSurvLoss, nll_loss
from util import cindex


class Normalize(object):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        assert len(self.mean)==len(self.std), "mismatch between size of mean and size of sed"
    def __call__(self, tensors):
        result = []
        assert len(self.mean)==len(tensors), "mismatch between size of mean and size of tensor"
        for i in range(len(tensors)):
            assert tensors[i].shape[-1] == self.mean[i].shape[-1] and tensors[i].shape[-1] == self.std[i].shape[-1], "prompt size should match size of mean and std"
            result.append((tensors[i]-self.mean[i].to(tensors[i].device))/self.std[i].to(tensors[i].device))
        return result

class GaussianNoise(object):
    def __init__(self, mean, std):
        super().__init__()
        if mean is not None and std is not None:
            self.norm = Normalize(mean, std)
        else:
            self.norm = None
    def __call__(self, tensors, weight, intensity):
        if self.norm is None:
            # print("Gaussian Noise stop working")
            return tensors
        result = []
        normed_tensors = self.norm(tensors)
        for i in range(len(tensors)):
            noise = torch.randn(normed_tensors[i].size()).to(tensors[i].device)*weight[i].to(tensors[i].device)*intensity[i]
            result.append(normed_tensors[i] + noise)
        return result


@ALGORITHMS.register('nsclccox_fusion')
class NSCLCCoxFusion(AlgorithmBase):
    """
        

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
            - ema_p (`float`):
                exponential moving average of probability update
        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(T=args.T, hard_label=args.hard_label, dist_align=args.dist_align, dist_uniform=args.dist_uniform, ema_p=args.ema_p, n_sigma=args.n_sigma, per_class=args.per_class)
    
    def init(self, T, hard_label=True, dist_align=True, dist_uniform=True, ema_p=0.999, n_sigma=2, per_class=False):
        self.T = T
        self.use_hard_label = hard_label
        self.dist_align = dist_align
        self.dist_uniform = dist_uniform
        self.ema_p = ema_p
        self.n_sigma = n_sigma
        self.per_class = per_class
        self.t_weight = [torch.tensor([1 for _ in range(52)])]###<---之后可能改成显著性的倒数
        self.nll_loss = nll_loss
        self.ce_surv_loss = CrossEntropySurvLoss()
        self.cox_loss = CoxPHSurvLoss()

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p, p_target_type='uniform' if self.args.dist_uniform else 'model'), 
            "DistAlignHook")
        self.register_hook(SoftMatchWeightingHook(num_classes=self.num_classes, n_sigma=self.args.n_sigma, momentum=self.args.ema_p, per_class=self.args.per_class), "MaskingHook")
        super().set_hooks()    

    def train_step(self, x_lb, y_lb, t_lb, x_ulb_w, x_ulb_s, t_ulb, y_ulb):
        # for param in self.model.named_parameters():
        #     print(param[0])
        if self.args.freeze_backbone:
            for param in self.model.named_parameters():
                if param[0].startswith('module.blocks') and not param[0].find('mona'):
                    param[1].requires_grad = False
        t_weak_intensity = [0.1]
        t_strong_intensity = [1]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            t_lb_w = self.t_transform(t_lb,self.t_weight,t_weak_intensity)
            t_ulb_w = self.t_transform(t_ulb,self.t_weight,t_weak_intensity)
            t_ulb_s = self.t_transform(t_ulb,self.t_weight,t_strong_intensity)
            # if self.args.prompt_split:
            #     t_lb_w = t_lb_w.permute(0,2,1)
            #     t_ulb_w = t_ulb_w.permute(0,2,1)
            #     t_ulb_s = t_ulb_s.permute(0,2,1)
            #     t_lb_w = [t_lb_w[:,t:t+1,:] for t in range(len(t_lb_w[0]))]
            #     t_ulb_w = [t_ulb_w[:,t:t+1,:] for t in range(len(t_ulb_w[0]))]
            #     t_ulb_s = [t_ulb_s[:,t:t+1,:] for t in range(len(t_ulb_s[0]))]
            # else:
            #     t_lb_w = [t_lb_w]
            #     t_ulb_w = [t_ulb_w]
            #     t_ulb_s = [t_ulb_s]
            outs_x_lb = self.model(x_lb, t_lb_w) 
            logits_x_lb = outs_x_lb['logits']
            feats_x_lb = outs_x_lb['feat']
            outs_x_ulb_s = self.model(x_ulb_s, t_ulb_s)
            logits_x_ulb_s = outs_x_ulb_s['logits']
            feats_x_ulb_s = outs_x_ulb_s['feat']
            with torch.no_grad():
                outs_x_ulb_w = self.model(x_ulb_w, t_ulb_w)
                logits_x_ulb_w = outs_x_ulb_w['logits']
                feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}


            # sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            if self.args.surv_loss == "ce":
                # hazards = torch.softmax(logits_x_lb, dim=-1)
                # sup_loss = self.ce_loss(hazards=hazards, survival=None, y_disc=y_lb, 
                #                         censorship=torch.zeros((self.args.batch_size)).to(y_lb.device))
                hazards = torch.softmax(torch.cat((logits_x_lb,logits_x_ulb_w),dim=0), dim=-1)
                sup_loss = self.ce_surv_loss(hazards=hazards, survival=None, 
                                         y_disc=torch.cat([y_lb,y_ulb],dim=0),
                                         censorship=torch.cat([torch.zeros([self.args.batch_size]),torch.ones([self.args.batch_size])],dim=0).to(hazards.device))
            elif self.args.surv_loss == "nll":
                # hazards = torch.softmax(logits_x_lb, dim=-1)
                # sup_loss = self.nll_loss(hazards=hazards, S=None, Y=y_lb,
                #                          c=torch.zeros((self.args.batch_size)).to(y_lb.device))
                hazards = torch.softmax(torch.cat((logits_x_lb,logits_x_ulb_w),dim=0), dim=-1)
                sup_loss = self.nll_loss(hazards=hazards, S=None, 
                                         Y=torch.cat([y_lb,y_ulb],dim=0),
                                         c=torch.cat([torch.zeros([self.args.batch_size]),torch.ones([self.args.batch_size])],dim=0).to(hazards.device))
            elif self.args.surv_loss == "cox":
                hazards = torch.softmax(torch.cat((logits_x_lb,logits_x_ulb_w),dim=0), dim=-1)
                survival = torch.cumprod(1 - hazards, dim=1) 
                sup_loss = self.cox_loss(hazards=hazards, survival=survival, 
                                         censorship=torch.cat([torch.zeros([self.args.batch_size]),torch.ones([self.args.batch_size])],dim=0).to(hazards.device), 
                                         device=hazards.device)

            probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)

            # uniform distribution alignment
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w, probs_x_lb=probs_x_lb)

            # calculate weight
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          # make sure this is logits, not dist aligned probs
                                          # uniform alignment in softmatch do not use aligned probs for generating pesudo labels
                                          logits=logits_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T)

            # calculate loss
            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                          pseudo_label,
                                          'ce',
                                          mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        """
        evaluation function
        """
        self.model.eval()
        self.ema.apply_shadow()

        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_probs = []
        y_logits = []
        with torch.no_grad():
            for data in eval_loader:
                x = data['x_lb']
                t = data['t_lb']
                y = data['y_lb']
                t_normed = self.t_transform(t,self.t_weight,0)
                # if self.args.prompt_split:
                #     t_normed = t_normed.permute(0, 2, 1)
                #     t_normed = [t_normed[:,t,:] for t in range(len(t_normed[0]))]
                # else:
                #     t_normed = [t_normed]
                
                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                elif isinstance(x,list):
                    x = [xi.cuda(self.gpu) for xi in x]
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                logits = self.model(x,t_normed)[out_key]
                
                loss = F.cross_entropy(logits, y, reduction='mean', ignore_index=-1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(logits.cpu().numpy())
                y_probs.extend(torch.softmax(logits, dim=-1).cpu().tolist())
                total_loss += loss.item() * num_batch
        #max_values, max_indexs = torch.max(y_pred, dim=-1)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)

        errors = abs(y_pred - y_true)
        mape = 100*(errors/y_true)
        accuracy = 100 - np.mean(mape)
        cindexs = cindex(score=y_pred,gt=y_true)
        eval_dict = {'\n'+eval_dest+'/loss': total_loss / total_num, eval_dest+"/MAE": round(np.mean(errors), 2),
                     eval_dest+"/top-1-acc": round(accuracy,2), eval_dest+"/C-index": round(cindexs,2)}
        # eval_dict = {'\n'+eval_dest+'/loss': total_loss / total_num, eval_dest+'/precision': precision, eval_dest+'/recall': recall, eval_dest+'/F1': F1,
        #              eval_dest+'/top-1-acc': top1, eval_dest+'/top-5-acc': top5, eval_dest+'/balanced_acc': balanced_top1, 
        #              eval_dest+'/AUC':auc_value, eval_dest+'/Classification Report':report}
        if return_logits:
            eval_dict[eval_dest+'/logits'] = y_logits
        # attn_weight = self.model.module.get_attention_weights()
        # for i, weight in enumerate(attn_weight):
        #     eval_dict[eval_dest+f'/attn{i}'] = weight
        return eval_dict

    def set_model(self):
        """
        initialize model
        """
        model = self.net_builder(num_classes=self.num_classes, fusing=self.args.fusing,
                                 fourier_encode_data = False,
                                 pretrained=self.args.use_pretrain, pretrained_path=self.args.pretrain_path,pretrained_from=self.args.pretrained_from)
        return model

    def set_ema_model(self):
        """
        initialize ema model from model
        """
        ema_model = self.net_builder(num_classes=self.num_classes, fusing=self.args.fusing, fourier_encode_data = False,)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def set_dataset(self):
        """
        set dataset_dict
        """
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        from semilearn.datasets import get_nsclc
        lb_dset, ulb_dset, eval_dset, mean, std = get_nsclc(args=self.args, alg=self.algorithm, include_lb_to_ulb=True)
        test_dset = None
        dataset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset, 'test': test_dset}
        self.t_transform = GaussianNoise(mean,std)
        if dataset_dict is None:
            return dataset_dict

        self.args.ulb_dest_len = len(dataset_dict['train_ulb']) if dataset_dict['train_ulb'] is not None else 0
        self.args.lb_dest_len = len(dataset_dict['train_lb'])
        self.print_fn("unlabeled data number: {}, labeled data number {}".format(self.args.ulb_dest_len, self.args.lb_dest_len))
        if self.rank == 0 and self.distributed:
            torch.distributed.barrier()
        return dataset_dict
    
    def set_data_loader(self):
        """
        set loader_dict
        """
        if self.dataset_dict is None:
            return
            
        self.print_fn("Create train and test data loaders")
        sampler_weights = [float(i) for i in self.args.sampler_weights.split(',')]
        loader_dict = {}
        loader_dict['train_lb'] = get_data_loader(self.args,
                                                  self.dataset_dict['train_lb'],
                                                  self.args.batch_size,
                                                  data_sampler=self.args.train_sampler,
                                                  sampler_weights=sampler_weights,
                                                  num_iters=self.num_train_iter,
                                                  num_epochs=self.epochs,
                                                  num_workers=self.args.num_workers,
                                                  distributed=self.distributed)

        loader_dict['train_ulb'] = get_data_loader(self.args,
                                                   self.dataset_dict['train_ulb'],
                                                   self.args.batch_size * self.args.uratio,
                                                   data_sampler=self.args.train_sampler,
                                                   sampler_weights=sampler_weights,
                                                   num_iters=self.num_train_iter,
                                                   num_epochs=self.epochs,
                                                   num_workers=2 * self.args.num_workers,
                                                   distributed=self.distributed)

        loader_dict['eval'] = get_data_loader(self.args,
                                              self.dataset_dict['eval'],
                                              self.args.eval_batch_size,
                                              # make sure data_sampler is None for evaluation
                                              data_sampler=None,
                                              num_workers=self.args.num_workers,
                                              drop_last=False)
        
        if self.dataset_dict['test'] is not None:
            loader_dict['test'] =  get_data_loader(self.args,
                                                   self.dataset_dict['test'],
                                                   self.args.eval_batch_size,
                                                   # make sure data_sampler is None for evaluation
                                                   data_sampler=None,
                                                   num_workers=self.args.num_workers,
                                                   drop_last=False)
        self.print_fn(f'[!] data loader keys: {loader_dict.keys()}')
        return loader_dict

    # TODO: change these
    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
        save_dict['prob_max_mu_t'] = self.hooks_dict['MaskingHook'].prob_max_mu_t.cpu()
        save_dict['prob_max_var_t'] = self.hooks_dict['MaskingHook'].prob_max_var_t.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].prob_max_mu_t = checkpoint['prob_max_mu_t'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].prob_max_var_t = checkpoint['prob_max_var_t'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--dist_align', str2bool, True),
            SSL_Argument('--dist_uniform', str2bool, True),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--n_sigma', int, 2),
            SSL_Argument('--per_class', str2bool, False),
        ]

import torch.nn as nn
from semilearn.core.criterions import ce_loss
class ConsistencyLoss(nn.Module):
    """
    Wrapper for consistency loss
    """
    def forward(self, logits, targets, name='ce', mask=None):
        """
        wrapper for consistency regularization loss in semi-supervised learning.

        Args:
            logits: logit to calculate the loss on and back-propagion, usually being the strong-augmented unlabeled samples
            targets: pseudo-labels (either hard label or soft label)
            name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
            mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
        """

        assert name in ['ce', 'mse', 'kl']
        # logits_w = logits_w.detach()
        if name == 'mse':
            probs = torch.softmax(logits, dim=-1)
            loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
        elif name == 'kl':
            loss = F.kl_div(F.log_softmax(logits / 0.5, dim=-1), F.softmax(targets / 0.5, dim=-1), reduction='none')
            loss = torch.sum(loss * (1.0 - mask).unsqueeze(dim=-1).repeat(1, torch.softmax(logits, dim=-1).shape[1]), dim=1)
        else:
            loss = ce_loss(logits, targets, reduction='none')

        if mask is not None and name != 'kl':
            # mask must not be boolean type
            loss = loss * mask

        return loss.mean()