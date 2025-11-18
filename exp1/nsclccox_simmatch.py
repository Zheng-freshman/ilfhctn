# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS, get_data_loader
from semilearn.algorithms.hooks import DistAlignQueueHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, concat_all_gather
from semilearn.core.criterions import CrossEntropySurvLoss, CoxPHSurvLoss, nll_loss
import numpy as np
from util import cindex


class SimMatch_Net(nn.Module):
    def __init__(self, base, proj_size=128, epass=False):
        super(SimMatch_Net, self).__init__()
        self.backbone = base
        self.epass = epass
        self.num_features = base.num_features
        
        self.mlp_proj = nn.Sequential(*[
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(inplace=False),
            nn.Linear(self.num_features, proj_size)
        ])
        
        if self.epass:
            self.mlp_proj_2 = nn.Sequential(*[
                nn.Linear(self.num_features, self.num_features),
                nn.ReLU(inplace=False),
                nn.Linear(self.num_features, proj_size)
            ])
            
            self.mlp_proj_3 = nn.Sequential(*[
                nn.Linear(self.num_features, self.num_features),
                nn.ReLU(inplace=False),
                nn.Linear(self.num_features, proj_size)
            ])
            
    def l2norm(self, x, power=2):
        norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
        out = x.div(norm)
        return out
    
    def forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        if self.epass:
            feat_proj = self.l2norm((self.mlp_proj(feat) + self.mlp_proj_2(feat) + self.mlp_proj_3(feat))/3)
        else:
            feat_proj = self.l2norm(self.mlp_proj(feat))
        return {'logits':logits, 'feat':feat_proj}

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher

@ALGORITHMS.register('nsclccox_simmatch')
class NSCLCCoxSimMatch(AlgorithmBase):
    """
    SimMatch algorithm (https://arxiv.org/abs/2203.06915).
    Reference implementation (https://github.com/KyleZheng1997/simmatch).

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
        - K (`int`, *optional*, default to 128):
            Length of the memory bank to store class probabilities and embeddings of the past weakly augmented samples
        - smoothing_alpha (`float`, *optional*, default to 0.999):
            Weight for a smoothness constraint which encourages taking a similar value as its nearby samples’ class probabilities
        - da_len (`int`, *optional*, default to 256):
            Length of the memory bank for distribution alignment.
        - in_loss_ratio (`float`, *optional*, default to 1.0):
            Loss weight for simmatch feature loss
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # simmatch specified arguments
        # adjust k 
        self.use_ema_teacher = True
        if args.dataset in ['stl10', 'cifar10', 'cifar100', 'svhn', 'superks', 'tissuemnist', 'eurosat', 'superbks', 'esc50', 'gtzan', 'urbansound8k', 'aclImdb', 'ag_news', 'dbpedia']:
            self.use_ema_teacher = False
            self.ema_bank = 0.7
        args.K = args.lb_dest_len
        self.lambda_in = args.in_loss_ratio
        self.init(T=args.T, p_cutoff=args.p_cutoff, proj_size=args.proj_size, K=args.K, smoothing_alpha=args.smoothing_alpha, da_len=args.da_len)
    

    def init(self, T, p_cutoff, proj_size, K, smoothing_alpha, da_len=0):
        self.T = T 
        self.p_cutoff = p_cutoff
        self.proj_size = proj_size 
        self.K = K
        self.smoothing_alpha = smoothing_alpha
        self.da_len = da_len

        # TODO：move this part into a hook
        # memory bank
        self.mem_bank = torch.randn(proj_size, K).cuda(self.gpu)
        self.mem_bank = F.normalize(self.mem_bank, dim=0)
        self.labels_bank = torch.zeros(K, dtype=torch.long).cuda(self.gpu)

    def set_hooks(self):
        self.register_hook(
            DistAlignQueueHook(num_classes=self.num_classes, queue_length=self.args.da_len, p_target_type='uniform'), 
            "DistAlignHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def set_model(self): 
        model = super().set_model()
        model = SimMatch_Net(model, proj_size=self.args.proj_size, epass=self.args.use_epass)
        return model
    
    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = SimMatch_Net(ema_model, proj_size=self.args.proj_size, epass=self.args.use_epass)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model    

    def set_dataset(self):
        """
        set dataset_dict
        """
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        from semilearn.datasets import get_nsclc
        lb_dset, ulb_dset, eval_dset, mean, std = get_nsclc(args=self.args, alg=self.algorithm, include_lb_to_ulb=True, target_type=self.args.target_type)
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


    @torch.no_grad()
    def update_bank(self, k, labels, index):
        if self.distributed and self.world_size > 1:
            k = concat_all_gather(k)
            labels = concat_all_gather(labels)
            index = concat_all_gather(index)
        if self.use_ema_teacher:
            self.mem_bank[:, index] = k.t().detach()
        else:
            self.mem_bank[:, index] = F.normalize(self.ema_bank * self.mem_bank[:, index] + (1 - self.ema_bank) * k.t().detach())
        self.labels_bank[index] = labels.detach()
    
    def train_step(self, idx_lb, x_lb, y_lb, x_ulb_w, x_ulb_s, y_ulb):
        num_lb = y_lb.shape[0]
        num_ulb = len(x_ulb_w['input_ids']) if isinstance(x_ulb_w, dict) else x_ulb_w[0].shape[0]
        idx_lb = idx_lb.cuda(self.gpu)

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            bank = self.mem_bank.clone().detach()

            if self.use_cat:
                # inputs = torch.cat((x_lb, x_ulb_s))
                # logits, feats = self.model(inputs)
                # logits_x_lb, ema_feats_x_lb = logits[:num_lb], feats[:num_lb]
                # logits_x_ulb_s, feats_x_ulb_s = logits[num_lb:], feats[num_lb:]
#######################################2
                # inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                inputs=[]
                for x1,x2,x3 in zip(x_lb,x_ulb_w,x_ulb_s):
                    inputs.append(torch.cat((x1,x2,x3)))
                outputs = self.model(inputs)
                logits, feats = outputs['logits'], outputs['feat']
                # logits, feats = self.model(inputs)
                logits_x_lb, ema_feats_x_lb = logits[:num_lb], feats[:num_lb]
                ema_logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                ema_feats_x_ulb_w, feats_x_ulb_s = feats[num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb)
                logits_x_lb, ema_feats_x_lb  = outs_x_lb['logits'], outs_x_lb['feat']
                # logits_x_lb, ema_feats_x_lb = self.model(x_lb)

                outs_x_ulb_w = self.model(x_ulb_w)
                ema_logits_x_ulb_w, ema_feats_x_ulb_w = outs_x_ulb_w['logits'], outs_x_ulb_w['feat']
                # ema_logits_x_ulb_w, ema_feats_x_ulb_w = self.model(x_ulb_w)

                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s, feats_x_ulb_s = outs_x_ulb_s['logits'], outs_x_ulb_s['feat']
                # logits_x_ulb_s, feats_x_ulb_s = self.model(x_ulb_s)

            # sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            if self.args.surv_loss == "ce":
                # hazards = torch.softmax(logits_x_lb, dim=-1)
                # sup_loss = self.ce_loss(hazards=hazards, survival=None, y_disc=y_lb, 
                #                         censorship=torch.zeros((self.args.batch_size)).to(y_lb.device))
                hazards = torch.softmax(torch.cat((logits_x_lb,ema_logits_x_ulb_w),dim=0), dim=-1)
                sup_loss = CrossEntropySurvLoss(hazards=hazards, survival=None, 
                                         y_disc=torch.cat([y_lb,y_ulb],dim=0),
                                         censorship=torch.cat([torch.zeros([self.args.batch_size]),torch.ones([self.args.batch_size])],dim=0).to(hazards.device))
            elif self.args.surv_loss == "nll":
                # hazards = torch.softmax(logits_x_lb, dim=-1)
                # sup_loss = self.nll_loss(hazards=hazards, S=None, Y=y_lb,
                #                          c=torch.zeros((self.args.batch_size)).to(y_lb.device))
                hazards = torch.softmax(torch.cat((logits_x_lb,ema_logits_x_ulb_w),dim=0), dim=-1)
                sup_loss = nll_loss(hazards=hazards, S=None, 
                                         Y=torch.cat([y_lb,y_ulb],dim=0),
                                         c=torch.cat([torch.zeros([self.args.batch_size]),torch.ones([self.args.batch_size])],dim=0).to(hazards.device))
            elif self.args.surv_loss == "cox":
                hazards = torch.softmax(torch.cat((logits_x_lb,ema_logits_x_ulb_w),dim=0), dim=-1)
                survival = torch.cumprod(1 - hazards, dim=1) 
                sup_loss = CoxPHSurvLoss(hazards=hazards, survival=survival, 
                                         censorship=torch.cat([torch.zeros([self.args.batch_size]),torch.ones([self.args.batch_size])],dim=0).to(hazards.device), 
                                         device=hazards.device)

            self.ema.apply_shadow()
            with torch.no_grad():
                # ema teacher model
                if self.use_ema_teacher:
                    ema_feats_x_lb = self.model(x_lb)['feat']
                ema_probs_x_ulb_w = F.softmax(ema_logits_x_ulb_w, dim=-1)
                ema_probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=ema_probs_x_ulb_w.detach())
            self.ema.restore()
            feat_dict = {'x_lb': ema_feats_x_lb, 'x_ulb_w':ema_feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            with torch.no_grad():
                teacher_logits = ema_feats_x_ulb_w @ bank
                teacher_prob_orig = F.softmax(teacher_logits / self.T, dim=1)
                factor = ema_probs_x_ulb_w.gather(1, self.labels_bank.expand([num_ulb, -1]))    
                teacher_prob = teacher_prob_orig * factor
                teacher_prob /= torch.sum(teacher_prob, dim=1, keepdim=True)

                if self.smoothing_alpha < 1:
                    bs = teacher_prob_orig.size(0)
                    aggregated_prob = torch.zeros([bs, self.num_classes], device=teacher_prob_orig.device)
                    aggregated_prob = aggregated_prob.scatter_add(1, self.labels_bank.expand([bs,-1]) , teacher_prob_orig)
                    probs_x_ulb_w = ema_probs_x_ulb_w * self.smoothing_alpha + aggregated_prob * (1- self.smoothing_alpha)
                else:
                    probs_x_ulb_w = ema_probs_x_ulb_w

            student_logits = feats_x_ulb_s @ bank
            student_prob = F.softmax(student_logits / self.T, dim=1)
            in_loss = torch.sum(-teacher_prob.detach() * torch.log(student_prob), dim=1).mean()
            if self.epoch == 0:
                in_loss *= 0.0
                probs_x_ulb_w = ema_probs_x_ulb_w

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               probs_x_ulb_w,
                                               'ce',
                                               mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_in * in_loss

            self.update_bank(ema_feats_x_lb, y_lb, idx_lb)

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
                y = data['y_lb']
                
                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
#######################################3
                elif isinstance(x,list):
                    x = [xi.cuda(self.gpu) for xi in x]
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
        save_dict['mem_bank'] = self.mem_bank.cpu()
        save_dict['labels_bank'] = self.labels_bank.cpu()
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu() 
        save_dict['p_model_ptr'] = self.hooks_dict['DistAlignHook'].p_model_ptr.cpu()
        return save_dict
    
    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.mem_bank = checkpoint['mem_bank'].cuda(self.gpu)
        self.labels_bank = checkpoint['labels_bank'].cuda(self.gpu)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_model_ptr = checkpoint['p_model_ptr'].cuda(self.args.gpu)
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--proj_size', int, 128),
            SSL_Argument('--K', int, 128),
            SSL_Argument('--in_loss_ratio', float, 1.0),
            SSL_Argument('--smoothing_alpha', float, 0.9),
            SSL_Argument('--da_len', int, 256),
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