# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/evaluation.py

import os
from .hook import Hook


class EvaluationHook(Hook):
    """
    Evaluation Hook for validation during training
    """
    
    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            algorithm.print_fn("validating...")
            eval_dict = algorithm.evaluate('eval')
            algorithm.log_dict.update(eval_dict)

            # update best metrics
            if 'eval/top-1-acc' in algorithm.log_dict:
                if algorithm.log_dict['eval/top-1-acc'] > algorithm.best_eval_acc:
                    algorithm.best_eval_acc = algorithm.log_dict['eval/top-1-acc']
                    algorithm.best_it = algorithm.it
            elif 'eval/top-1-acc_1' in algorithm.log_dict:
                if not hasattr(algorithm, "best_eval_acc_list") or not hasattr(algorithm, "best_it_list"):
                    algorithm.best_eval_acc_list = [0,0]
                    algorithm.best_it_list = [0,0]
                if algorithm.log_dict['eval/top-1-acc_1'] > algorithm.best_eval_acc_list[0]:
                    algorithm.best_eval_acc_list[0] = algorithm.log_dict['eval/top-1-acc_1']
                    algorithm.best_it_list[0] = algorithm.it
                if algorithm.log_dict['eval/top-1-acc_2'] > algorithm.best_eval_acc_list[1]:
                    algorithm.best_eval_acc_list[1] = algorithm.log_dict['eval/top-1-acc_2']
                    algorithm.best_it_list[1] = algorithm.it
                if algorithm.log_dict['eval/top-1-acc_1']+algorithm.log_dict['eval/top-1-acc_2'] > algorithm.best_eval_acc:
                    algorithm.best_eval_acc = algorithm.log_dict['eval/top-1-acc_1']+algorithm.log_dict['eval/top-1-acc_2']
                    algorithm.best_it = algorithm.it
    
    def after_run(self, algorithm):
        
        if not algorithm.args.multiprocessing_distributed or (algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
            save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
            algorithm.save_model('latest_model.pth', save_path)

        if not hasattr(algorithm, "best_eval_acc_list"): 
            results_dict = {'eval/best_acc': algorithm.best_eval_acc, 'eval/best_it': algorithm.best_it}
        else:
            results_dict = {'eval/best_acc_1': algorithm.best_eval_acc_list[0], 'eval/best_acc_2': algorithm.best_eval_acc_list[1], 
                            'eval/best_it_1': algorithm.best_it_list[0], 'eval/best_it_2': algorithm.best_it_list[1]}
        if 'test' in algorithm.loader_dict:
            # load the best model and evaluate on test dataset
            best_model_path = os.path.join(algorithm.args.save_dir, algorithm.args.save_name, 'model_best.pth')
            algorithm.load_model(best_model_path)
            test_dict = algorithm.evaluate('test')
            results_dict['test/best_acc'] = test_dict['test/top-1-acc']
        algorithm.results_dict = results_dict
        