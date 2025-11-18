from torch.utils.data import Dataset
from torchvision import transforms
#import cv2
from PIL import Image
import os
import numpy
from semilearn.datasets.utils import get_onehot
import torch


class BasicDataset(Dataset):
    def __init__(self, 
                 NO_list,
                 dataset_path, 
                 get_meta, 
                 info_data, 
                 alg,
                 gray=False, 
                 image_pathlist=["1198/GRAYlspine", "1198/GRAYhip1"],
                 suffix_list=["lspine.jpg","hip1.jpg"],
                 transform=None,
                 is_ulb=False,
                 medium_transform=None,
                 strong_transform=None,
                 onehot=False,
                 t_norm=None,
                 verbose=True):
        """
        :image_path  directory of Image
        :get_meta function for get prompt and target from Pandas Sheet
        get_meta(pandas.dataframe, NO)->prompt_list, target_list
        :info_path position of the Excel
        :page "1198"/"294"
        :screen screen Class judge if image should be ignore
        screen.filtrarate(image_file_name, pandas.dataframe)-->return True when pass
        """
        self.image_pathlist = []
        for image_path in image_pathlist:
            self.image_pathlist.append(os.path.join(dataset_path, image_path))
        self.suffix_list = suffix_list
        self.get_meta = get_meta
        self.NO_list=[]
        self.targets = []
        self.sheet = info_data
        self.gray = gray

        self.alg = alg
        self.transform = transform
        self.is_ulb = is_ulb
        self.medium_transform = medium_transform
        self.strong_transform = strong_transform
        self.onehot = onehot
        self.num_classes = 2

        self.NO_list = NO_list
        targets = []
        for NO in NO_list:
            meta = self.get_meta(NO=NO,df=self.sheet)
            if isinstance(meta["target"], list) and len(meta["target"])==1:
                targets.append(int(meta['target'][0]))
            else:
                targets.append(int(meta['target']))
        self.targets = targets

        # reject caseID in Ban List or not in sheet
        # NO_form=[]
        # for image_path in image_pathlist: #[lspine_path,hip1_path]
        #     sec_list=[]
        #     for imgName in os.listdir(os.path.join(dataset_path, image_path)):
        #         NO,suffix = imgName.split('.', maxsplit=1)
        #         if rejecter is None or rejecter.filtrate(imgName, self.sheet):
        #             sec_list.append(NO)
        #     self.suffix_list.append(suffix)
        #     NO_form.append(sec_list)
        # for NO in NO_form[0]:
        #     full = True
        #     for sec_list in NO_form[1:]:
        #         meta = self.get_meta(NO=NO,df=self.sheet)
        #         if NO not in sec_list and meta is not None:
        #             full = False
        #     if full:
        #         self.NO_list.append(NO)
        #         if isinstance(meta["target"], list) and len(meta["target"])==1:
        #             self.targets += meta['target']
        #         else:
        #             self.targets.append(meta['target'])
        # print("DATASET: data inited")
    
    def __len__(self):
        return len(self.NO_list)
    
    def __getitem__(self, index):
        imageNO = self.NO_list[index]
        meta = self.get_meta(NO=imageNO, df=self.sheet)
        prompt_list = meta["prompt"]
        target_list = meta["target"]
        NO = meta["prompt"]
        img_list=[]
        for set_it, data in enumerate(zip(self.image_pathlist, self.suffix_list)):
            image_path, suffix = data
            path = os.path.join(image_path,".".join((imageNO,suffix)))
            if self.gray:
                img = Image.open(path).convert("L")
            else:
                img = Image.open(path).convert("RGB")
            img = numpy.array(img)
            img_list.append(img)
        
        img = img_list[1] #####<--兼容非融合算法
        text = torch.tensor(prompt_list)
        if target_list is None:
            target = None
        else:
            target_ = target_list[0]
            target = torch.tensor(target_).long() if not self.onehot else get_onehot(self.num_classes, target_)
        if self.transform is None:
            # if self.alg == 'sup_healnet':
            #     return {'x_lb': img_list_w, 
            #             'y_lb': target, 
            #             't_lb': [torch.tensor(t) for t in prompt_list], 
            #             'frax_lb':torch.tensor(meta["FRAXs"])}
            # else:
            #     return  {'x_lb': transforms.ToTensor()(img), 
            #             'y_lb': target, 
            #             't_lb': text, 
            #             'frax_lb':torch.tensor(meta["FRAXs"])}
            return  {'x_lb': transforms.ToTensor()(img), 
                    'y_lb': target, 
                    't_lb': [torch.tensor(t) for t in prompt_list], 
                    'frax_lb':torch.tensor(meta["FRAXs"])}
        else:
            if isinstance(img, numpy.ndarray):
                img_list = [Image.fromarray(x) for x in img_list]
                img = img_list[1] ###<-取髋部图
            img_w = self.transform(img)
            img_list_w = [self.transform(x) for x in img_list]
            if not self.is_ulb:
                # if self.alg == 'sup_healnet' or self.alg == 'softmatch_fusion' or self.alg == 'softmatch_fusion_cox' or self.alg == 'softmatch_fusion_cox_dual':
                #     return {'idx_lb': index, 'x_lb': img_list_w, 
                #             'y_lb': target, 't_lb': [torch.tensor(t) for t in prompt_list], 
                #             'frax_lb':torch.tensor(meta["FRAXs"])}
                # else:
                #     return {'idx_lb': index, 'x_lb': img_w, 
                #             'y_lb': target, 't_lb': [torch.tensor(t) for t in prompt_list], 
                #             'frax_lb':torch.tensor(meta["FRAXs"])}
                return {'idx_lb': index, 'x_lb': img_list_w, 
                            'y_lb': target, 't_lb': [torch.tensor(t) for t in prompt_list], 
                            'frax_lb':torch.tensor(meta["FRAXs"])}
            else:
                if self.alg == 'fullysupervised' or self.alg == 'supervised':
                    return {'idx_ulb': index}
                if self.alg == 'sup_healnet':
                    return {'idx_ulb': index, 
                            'x_ulb': img_list_w,
                            't_ulb': [torch.tensor(t) for t in prompt_list],
                            'y_ulb': target,
                            'frax_ulb':torch.tensor(meta["FRAXs"])} 
                elif self.alg == 'softmatch_fusion':
                    return {'idx_ulb': index, 
                            'x_ulb_w': img_list_w, 'x_ulb_s': [self.strong_transform(x) for x in img_list], 
                            't_ulb': [torch.tensor(t) for t in prompt_list],
                            'frax_ulb':torch.tensor(meta["FRAXs"])} 
                elif self.alg == 'softmatch_fusion_cox' or self.alg == 'softmatch_fusion_cox_dual':
                    return {'idx_ulb': index, 
                            'x_ulb_w': img_list_w, 'x_ulb_s': [self.strong_transform(x) for x in img_list], 
                            't_ulb': [torch.tensor(t) for t in prompt_list], 'y_ulb': target,
                            'frax_ulb':torch.tensor(meta["FRAXs"])}
                elif self.alg=='simmatch_cox' or self.alg=='refixmatch_cox':
                    return {'idx_ulb': index, 
                            'x_ulb_w': img_list_w, 'x_ulb_s': [self.strong_transform(x) for x in img_list], 
                            't_ulb': [torch.tensor(t) for t in prompt_list], 'y_ulb': target,
                            'frax_ulb':torch.tensor(meta["FRAXs"])}
                elif self.alg=='sequencematch_cox':
                    return {'idx_ulb': index, 
                            'x_ulb_w': img_list_w, 'x_ulb_m': [self.medium_transform(x) for x in img_list], 'x_ulb_s': [self.strong_transform(x) for x in img_list], 
                            't_ulb': [torch.tensor(t) for t in prompt_list], 'y_ulb': target,
                            'frax_ulb':torch.tensor(meta["FRAXs"])}
                elif self.alg == 'pseudolabel' or self.alg == 'vat':
                    return {'idx_ulb': index, 'x_ulb_w':img_w} 
                elif self.alg == 'pimodel' or self.alg == 'meanteacher' or self.alg == 'mixmatch':
                    # NOTE x_ulb_s here is weak augmentation
                    return {'idx_ulb': index, 'x_ulb_w': img_w, 'x_ulb_s': self.transform(img)}
                # elif self.alg == 'sequencematch' or self.alg == 'somematch':
                elif self.alg == 'sequencematch':
                    return {'idx_ulb': index, 'x_ulb_w': img_w, 'x_ulb_m': self.medium_transform(img), 'x_ulb_s': self.strong_transform(img)} 
                elif self.alg == 'remixmatch':
                    rotate_v_list = [0, 90, 180, 270]
                    rotate_v1 = numpy.random.choice(rotate_v_list, 1).item()
                    img_s1 = self.strong_transform(img)
                    img_s1_rot = transforms.functional.rotate(img_s1, rotate_v1)
                    img_s2 = self.strong_transform(img)
                    return {'idx_ulb': index, 'x_ulb_w': img_w, 'x_ulb_s_0': img_s1, 'x_ulb_s_1':img_s2, 'x_ulb_s_0_rot':img_s1_rot, 'rot_v':rotate_v_list.index(rotate_v1)}
                elif self.alg == 'comatch':
                    return {'idx_ulb': index, 'x_ulb_w': img_w, 'x_ulb_s_0': self.strong_transform(img), 'x_ulb_s_1':self.strong_transform(img)} 
                else:
                    return {'idx_ulb': index, 'x_ulb_w': img_w, 'x_ulb_s': self.strong_transform(img)} 


class rejecter(object):
    def __init__(self):
        pass
    def filtrate(self, image_file_name, df):
        pass