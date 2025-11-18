# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS, get_data_loader
import torch
import numpy as np
from semilearn.core.criterions import CrossEntropySurvLoss, CoxPHSurvLoss, nll_loss
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

@ALGORITHMS.register('sup_healnet')
class FullySupervised(AlgorithmBase):
    """
        Train a fully supervised model using labeled data only. This serves as a baseline for comparison.

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)

    def train_step(self, x_lb, y_lb, t_lb, x_ulb, t_ulb, y_ulb):
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            x = [torch.cat([lb, ulb],dim=0) for lb,ulb in zip(x_lb, x_ulb)]
            t = [torch.cat([lb, ulb],dim=0) for lb,ulb in zip(t_lb, t_ulb)]
            logits_x = self.model(t+x)['logits']
            # logits_x_lb = self.model(t_lb+x_lb)['logits']
            # logits_x_ulb = self.model(t_ulb+x_ulb)['logits']
            # logits_x = torch.cat([logits_x_lb,logits_x_ulb], dim=0)
            censorship = torch.cat([torch.zeros([self.args.batch_size]),torch.ones([self.args.batch_size])],dim=0)
            hazards = torch.softmax(logits_x, dim=-1)
            # sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
        if self.args.surv_loss == "ce":
            # hazards = torch.softmax(logits_x_lb, dim=-1)
            # sup_loss = self.ce_loss(hazards=hazards, survival=None, y_disc=y_lb, 
            #                         censorship=torch.zeros((self.args.batch_size)).to(y_lb.device))
            
            sup_loss = CrossEntropySurvLoss(hazards=hazards, survival=None, 
                                   y_disc=torch.cat([y_lb,y_ulb],dim=0),
                                    censorship=censorship.to(hazards.device))
        elif self.args.surv_loss == "nll":
            # hazards = torch.softmax(logits_x_lb, dim=-1)
            # sup_loss = self.nll_loss(hazards=hazards, S=None, Y=y_lb,
            #                          c=torch.zeros((self.args.batch_size)).to(y_lb.device))
            sup_loss = nll_loss(hazards=hazards, S=None, 
                                    Y=torch.cat([y_lb,y_ulb],dim=0),
                                    c=censorship.to(hazards.device))
        elif self.args.surv_loss == "cox":
            survival = torch.cumprod(1 - hazards, dim=1) 
            sup_loss = CoxPHSurvLoss(hazards=hazards, survival=survival, 
                                    censorship=censorship.to(hazards.device), 
                                    device=hazards.device)

        out_dict = self.process_out_dict(loss=sup_loss)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item())
        return out_dict, log_dict

    
    def train(self):
        # lb: labeled, ulb: unlabeled
        self.model.train()
        self.call_hook("before_run")
            
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it > self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                         self.loader_dict['train_ulb']):

                # prevent the training iterations exceed args.num_train_iter
                if self.it > self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")
        self.call_hook("after_run")

    def evaluate(self, eval_dest='eval', out_key='logits', return_logits=False):
        """
        evaluation function
        """
        self.model.eval()

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
                
                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                if isinstance(x, list):
                    x = [v.cuda(self.gpu) for v in x]
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                outs_x_lb = self.model(t+x)
                logits = outs_x_lb[out_key]
                
                y_true.extend(y.cpu().tolist())
                # loss = F.cross_entropy(logits[i], y, reduction='mean', ignore_index=-1)
                hazards = torch.softmax(logits, dim=-1)
                sup_loss = nll_loss(hazards=hazards, S=None, Y=y,
                                        c=torch.ones([len(y)]).to(hazards.device))
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(logits.cpu().numpy())
                y_probs.extend(torch.softmax(logits, dim=-1).cpu().tolist())
                total_loss += sup_loss.item() * num_batch
        #max_values, max_indexs = torch.max(y_pred, dim=-1)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)

        errors = abs(y_pred - y_true)
        mape = 100*(errors/y_true)
        accuracy = 100 - np.mean(mape)
        c_index = cindex(score=y_pred,gt=y_true)

        eval_dict = {'\n'+eval_dest+'/sup_loss': total_loss / total_num,
                     '\n'+eval_dest+"/MAE": round(np.mean(errors), 2), eval_dest+"/top-1-acc": round(accuracy,2), eval_dest+"/C-index": round(c_index,2)}
        if return_logits:
            eval_dict[eval_dest+'/logits'] = y_logits
        return eval_dict

    def set_dataset(self):
        """
        set dataset_dict
        """
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        from semilearn.datasets import get_fracture_time
        lb_dset, ulb_dest, eval_dset, mean, std = get_fracture_time(args=self.args, alg=self.args.algorithm, include_lb_to_ulb=True, target_type=self.args.target_type)
        test_dset = None
        dataset_dict = {'train_lb': lb_dset, 'train_ulb':ulb_dest, 'eval': eval_dset, 'test': test_dset}
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
                               