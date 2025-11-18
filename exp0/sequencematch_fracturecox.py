
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from semilearn.algorithms.flexmatch.utils import FlexMatchThresholdingHook
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS, get_data_loader
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
import torch.nn.functional as F
from semilearn.core.criterions import CrossEntropySurvLoss, CoxPHSurvLoss, nll_loss
import numpy as np
from util import cindex


class Normalize(object):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        assert tensor.shape[-1] == self.mean.shape[-1] and tensor.shape[-1] == self.std.shape[-1], "prompt size should match size of mean and std"
        return (tensor-self.mean.to(tensor.device))/self.std.to(tensor.device)

class GaussianNoise(object):
    def __init__(self, mean, std):
        super().__init__()
        self.norm = Normalize(mean, std)
    def __call__(self, tensor, weight, intensity=0.1):
        normed_tensor = self.norm(tensor)
        noise = torch.randn(normed_tensor.size()).to(tensor.device)*weight.to(tensor.device)*intensity
        return normed_tensor + noise

@ALGORITHMS.register('sequencematch_cox')
class SequenceMatch(AlgorithmBase):
    """
        SequenceMatch algorithm (https://arxiv.org/abs/2110.08263).

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
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
            - ulb_dest_len (`int`):
                Length of unlabeled data
            - thresh_warmup (`bool`, *optional*, default to `True`):
                If True, warmup the confidence threshold, so that at the beginning of the training, all estimated
                learning effects gradually rise from 0 until the number of unused unlabeled data is no longer
                predominant

        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # flexmatch specificed arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label, thresh_warmup=args.thresh_warmup)
    
    def init(self, T, p_cutoff, hard_label=True, thresh_warmup=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.thresh_warmup = thresh_warmup

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FlexMatchThresholdingHook(ulb_dest_len=self.args.ulb_dest_len, num_classes=self.num_classes, thresh_warmup=self.args.thresh_warmup), "MaskingHook")
        super().set_hooks()

    def set_dataset(self):
        """
        set dataset_dict
        """
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        from semilearn.datasets import get_fracture_time
        lb_dset, ulb_dset, eval_dset, mean, std = get_fracture_time(args=self.args, alg=self.algorithm, include_lb_to_ulb=True, target_type=self.args.target_type)
        test_dset = None
        dataset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset, 'test': test_dset}
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

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_m, x_ulb_s, y_ulb):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_m, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                # feats_x_lb = outputs['feats'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_m, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(3)
                # feats_x_ulb_w, feats_x_ulb_m, feats_x_ulb_s = outputs['feats'][num_lb:].chunk(3)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                # feats_x_lb = outs_x_lb['feats']
                outs_x_ulb_m = self.model(x_ulb_m)
                logits_x_ulb_m = outs_x_ulb_m['logits']
                # feats_x_ulb_m = outs_x_ulb_m['feats']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                # feats_x_ulb_s = outs_x_ulb_s['feats']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
            # feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_m':feats_x_ulb_m, 'x_ulb_s':feats_x_ulb_s}

            # sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            if self.args.surv_loss == "ce":
                # hazards = torch.softmax(logits_x_lb, dim=-1)
                # sup_loss = self.ce_loss(hazards=hazards, survival=None, y_disc=y_lb, 
                #                         censorship=torch.zeros((self.args.batch_size)).to(y_lb.device))
                hazards = torch.softmax(torch.cat((logits_x_lb,logits_x_ulb_w),dim=0), dim=-1)
                sup_loss = CrossEntropySurvLoss(hazards=hazards, survival=None, 
                                         y_disc=torch.cat([y_lb,y_ulb],dim=0),
                                         censorship=torch.cat([torch.zeros([self.args.batch_size]),torch.ones([self.args.batch_size])],dim=0).to(hazards.device))
            elif self.args.surv_loss == "nll":
                # hazards = torch.softmax(logits_x_lb, dim=-1)
                # sup_loss = self.nll_loss(hazards=hazards, S=None, Y=y_lb,
                #                          c=torch.zeros((self.args.batch_size)).to(y_lb.device))
                hazards = torch.softmax(torch.cat((logits_x_lb,logits_x_ulb_w),dim=0), dim=-1)
                sup_loss = nll_loss(hazards=hazards, S=None, 
                                         Y=torch.cat([y_lb,y_ulb],dim=0),
                                         c=torch.cat([torch.zeros([self.args.batch_size]),torch.ones([self.args.batch_size])],dim=0).to(hazards.device))
            elif self.args.surv_loss == "cox":
                hazards = torch.softmax(torch.cat((logits_x_lb,logits_x_ulb_w),dim=0), dim=-1)
                survival = torch.cumprod(1 - hazards, dim=1) 
                sup_loss = CoxPHSurvLoss(hazards=hazards, survival=survival, 
                                         censorship=torch.cat([torch.zeros([self.args.batch_size]),torch.ones([self.args.batch_size])],dim=0).to(hazards.device), 
                                         device=hazards.device)

            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            probs_x_ulb_m = self.compute_prob(logits_x_ulb_m.detach())

            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())
                probs_x_ulb_m = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_m.detach())

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False, idx_ulb=idx_ulb)
            
            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                            logits=probs_x_ulb_w,
                                            use_hard_label=self.use_hard_label,
                                            T=self.T,
                                            softmax=False)

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                                pseudo_label,
                                                'ce',
                                                mask=mask)

            unsup_loss_mw = F.kl_div(F.softmax(logits_x_ulb_m, dim=-1).log(),
                                    F.softmax(probs_x_ulb_w / self.T, dim=-1).detach(),
                                    reduction='none').sum(dim=1, keepdim=False)
            unsup_loss_mw = (unsup_loss_mw * mask).mean()
            
            unsup_loss_sm = F.kl_div(F.softmax(logits_x_ulb_s, dim=-1).log(),
                                    F.softmax(probs_x_ulb_m / self.T, dim=-1).detach(),
                                    reduction='none').sum(dim=1, keepdim=False)
            unsup_loss_sm = (unsup_loss_sm * mask).mean()
            
            unsup_loss_sw = F.kl_div(F.softmax(logits_x_ulb_s, dim=-1).log(),
                                    F.softmax(probs_x_ulb_w / self.T, dim=-1).detach(),
                                    reduction='none').sum(dim=1, keepdim=False)
            unsup_loss_sw = (unsup_loss_sw * mask).mean()
            
            total_loss = sup_loss + self.lambda_u * (unsup_loss + unsup_loss_mw + unsup_loss_sm + unsup_loss_sw)

        # out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        out_dict = self.process_out_dict(loss=total_loss)
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
                y = data['y_lb']
                
                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                logits = self.model(x)[out_key]
                
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
   

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['classwise_acc'] = self.hooks_dict['MaskingHook'].classwise_acc.cpu()
        save_dict['selected_label'] = self.hooks_dict['MaskingHook'].selected_label.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].classwise_acc = checkpoint['classwise_acc'].cuda(self.gpu)
        self.hooks_dict['MaskingHook'].selected_label = checkpoint['selected_label'].cuda(self.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--thresh_warmup', str2bool, True),
        ]