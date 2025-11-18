import os
import time
import math
import json
import numpy
import pandas as pd
import torch
from torchvision import transforms
from semilearn.datasets.augmentation import RandAugment
from .datasetbase import BasicDataset
from torch.utils.data import Subset, ConcatDataset, random_split
from functools import partial


ban = ["v88", "1161"]


def get_fracture(args, alg, include_lb_to_ulb=True, splitNum=5, target_type="ifMOF"):
    month = 74
    crop_size = args.img_size
    crop_ratio = args.crop_ratio
    split_id = args.split_id
    dataset_path = args.data_dir
    info_path = os.path.join(dataset_path,"1198+294.xlsx")
    excel = pd.ExcelFile(info_path)
    info_data = excel.parse("1198")
    image_pathlist = ["1198/GRAYlspine", "1198/GRAYhip1"]
    mean = torch.tensor(get_1198_meta(info_data,mean=True)['prompt'][0])
    std = torch.tensor(get_1198_meta(info_data,std=True)['prompt'][0])

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])

    transform_medium = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 5),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        # RandAugment(3, 5),
        RandAugment(1, 5),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize([crop_size,crop_size]),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])


    get_meta = partial(get_1198_meta, target_type=args.target_type)
    filter_lb = filter_1198_knowMOF_in_month(month,ban)
    filter_ulb = filter_1198_unknowMOF_in_month(month,ban)
    filter_all = filter_1198(ban)

    filter_list = [filter_lb, filter_ulb]
    NO_list = []
    suffix_list = []
    data_dict = [{'NO_list':[],'targets':[]} for _ in range(len(filter_list))]
    NO_Intersection = None
    for image_path in image_pathlist: #[lspine_path,hip1_path]
        sec_list=[]
        for imgName in os.listdir(os.path.join(dataset_path, image_path)):
            NO,suffix = imgName.split('.', maxsplit=1)
            sec_list.append(NO)
        suffix_list.append(suffix)
        if NO_Intersection is None:
            NO_Intersection = set(sec_list)
        else:
            NO_Intersection = NO_Intersection & set(sec_list)
    NO_Intersection = list(NO_Intersection)
    NO_Intersection.sort()
    for NO in NO_Intersection:
        meta = get_meta(NO=NO,df=info_data)
        if meta is not None:
            NO_list.append(NO)
            for i, filter in enumerate(filter_list):
                if filter.filtrate(NO, info_data):
                    data_dict[i]['NO_list'].append(NO)

    print("DATASET: data init")
    lb_NO = data_dict[0]['NO_list']
    ulb_NO = data_dict[1]['NO_list']
    spliter_lb = spliter(lb_NO, ID="lb", splitNum=splitNum, path=args.save_dir)
    spliter_ulb = spliter(ulb_NO, ID="ulb", splitNum=splitNum, path=args.save_dir)
    train_lb_NO, test_lb_NO = spliter_lb.get_split(split_id)
    train_ulb_NO, test_ulb_NO = spliter_ulb.get_split(split_id)
    if include_lb_to_ulb==True:
        train_ulb_NO = list(set(train_lb_NO) | set(train_ulb_NO))

    lb_dset = BasicDataset(
        NO_list=train_lb_NO,
        dataset_path=dataset_path,
        get_meta=get_meta,
        info_data=info_data,
        alg=alg,
        gray=False,
        image_pathlist=image_pathlist,
        suffix_list=suffix_list,
        transform=transform_weak,
        is_ulb=False,
        medium_transform=transform_strong,
        strong_transform=transform_strong,
        onehot=False,)
    ulb_dset = BasicDataset(
        NO_list=train_ulb_NO,
        dataset_path=dataset_path,
        get_meta=get_meta,
        info_data=info_data,
        alg=alg,
        gray=False,
        image_pathlist=image_pathlist,
        suffix_list=suffix_list,
        transform=transform_weak,
        is_ulb=True,
        medium_transform=transform_medium,
        strong_transform=transform_strong,
        onehot=False,)
    eval_dset = BasicDataset(
        NO_list=test_lb_NO,
        dataset_path=dataset_path,
        get_meta=get_meta,
        info_data=info_data,
        alg=alg,
        gray=False,
        image_pathlist=image_pathlist,
        suffix_list=suffix_list,
        transform=transform_val,
        is_ulb=False,
        medium_transform=None,
        strong_transform=None,
        onehot=False,)
    
    
    print("Dataset lb: {}".format(len(lb_dset)))
    print("Dataset ulb: {}".format(len(ulb_dset)))

    return lb_dset, ulb_dset, eval_dset, mean, std
        

def get_1198_meta(df, target_type=None, NO=None, mean=False, std=False):
    if mean:
        row = df.loc[1198]
    elif std:
        row = df.loc[1199]
    else:
        row = df.loc[df['NO']==int(NO)]
        if row.size <= 0:
            return None
        else:
            row = row.iloc[0]
    #ifFrac = row["是否MOF"].values[0]
    healthInfo = row.iloc[3:48].values.tolist() #年龄--高血压
    BMD_T = row.iloc[48:55].values.tolist() #股骨颈、总髋、腰椎的BMD与T值，T值分类
    FRAXs = row.iloc[55:62].values.tolist() #各FRAX评分
    legBMD = [row["股骨颈BMD"]]
    hipBMD = [row["总髋BMD"]]
    lspineBMD = [row["腰椎BMD"]]

    prompt = healthInfo + BMD_T
    fracTime = [row["any骨折时间月"]]
    ifMOF = [row["是否MOF"]]
    #FRAXs.append(fracTime)
    FRAXs = [x for x in FRAXs]
    frac = [fracTime[0],ifMOF[0]]
    NO = NO

    classfity_target = ifMOF
    regress_target_FRAX = FRAXs
    regress_target_time = fracTime
    regress_target = None
    if target_type=="ifMOF":
        target = classfity_target
    elif target_type=="fracTime":
        target = regress_target_time
    elif not (mean or std):
        raise RuntimeError("unknow target type in fracture dataset")
    else:
        target=None
    return {"prompt":[prompt], "target":target, "NO":NO, "ifMOF":ifMOF, "fracTime":fracTime, "FRAXs":FRAXs}


class filter_1198_knowMOF_in_month(object):
    def __init__(self, month, ban=[]):
        self.ban = ban
        self.month = month
    def filtrate(self, NO, df):
        """
        return True when item should be reserve
        """    
        row = df.loc[df['NO'].apply(str)==NO]
        if len(row)==0:
            #print("miss info: ",NO)
            return False
        if NO not in self.ban and row['是否MOF'].values[0]==1 and row['any骨折时间月'].values[0]<=self.month:
            return True
        elif NO not in self.ban and row['是否MOF'].values[0]==0 and row['any骨折时间月'].values[0]>=self.month:
            return True
        else:
            return False

class filter_1198_unknowMOF_in_month(object):
    def __init__(self, month, ban=[]):
        self.ban = ban
        self.month = month
    def filtrate(self, NO, df):
        """
        return True when item should be reserve
        """    
        row = df.loc[df['NO'].apply(str)==NO]
        if len(row)==0:
            #print("miss info: ",NO)
            return False
        if NO not in self.ban and row['是否MOF'].values[0]==1 and row['any骨折时间月'].values[0]>self.month:
            return True
        elif NO not in self.ban and row['是否MOF'].values[0]==0 and row['any骨折时间月'].values[0]<self.month:
            return True
        else:
            return False


class filter_1198(object):
    def __init__(self, ban=[]):
        self.ban = ban
    def filtrate(self, NO, df):
        """
        return True when item should be reserve
        """    
        row = df.loc[df['NO'].apply(str)==NO]
        if len(row)==0:
            #print("miss info: ",NO)
            return False
        if NO not in self.ban:
            return True
        else:
            return False


class spliter:
    def __init__(self, NO_list, ID="Default", path="./", shuffle=True, rewrite_file=False, splitNum=5):
        """
        :full_dataset 
        :filename save the split result
        :shuffle shuffle when it is True
        :update_file if cover with a new split.json
        :splitNum the ratio of testset
        """
        filename = f"split_Dataset-{ID}_splitNum-{splitNum}.json"
        filepath = os.path.join(path, filename)
        self.NO_list = NO_list
        self.splitNum = splitNum
        if os.path.exists(filepath) and not rewrite_file:
            with open(filepath, 'r') as f:
                self.splitlist = json.load(f)
        else:
            f = open(filepath, 'w')
            self.splitlist=[]
            dataset_size = len(NO_list)
            indices = list(range(dataset_size))
            split = int(numpy.floor(1/splitNum * dataset_size))
            if shuffle :
                numpy.random.seed(numpy.int64(time.time()))
                numpy.random.shuffle(indices)
            for i in range(splitNum):
                train_indices, test_indices = indices[:i*split]+indices[(i+1)*split:], indices[i*split:(i+1)*split]
                self.splitlist.append((train_indices, test_indices))
            json.dump(self.splitlist, f)

    def get_split_id(self, it=-1):
        if it>=0 and it < len(self.splitlist):
            return self.splitlist[it][0], self.splitlist[it][1]
        else:
            return self.splitlist

    def get_split(self, it=-1):
        if it>=0 and it < len(self.splitlist):
            train = []
            test = []
            for i in self.splitlist[it][0]:
                train.append(self.NO_list[i])
            for i in self.splitlist[it][1]:
                test.append(self.NO_list[i])
            return train, test
        else:
            all_split = []
            for it in self.splitNum:
                train = []
                test = []
                for i in self.splitlist[it][0]:
                    train.append(self.NO_list[i])
                for i in self.splitlist[it][1]:
                    test.append(self.NO_list[i])
                all_split.append((train,test))
            return all_split


if __name__ == "__main__":
    ban = ["v88", "1161"]

    dataset_path = "../dataset/"
    info_path = "../dataset/1198+294.xlsx"
    page = "1198"
    excel = pd.ExcelFile(info_path)
    info_data = excel.parse(page) 

    set = BasicDataset(dataset_path=dataset_path,\
                    get_meta=get_1198_meta,\
                    info_data=info_data, rejecter=filter_1198_knowMOF_in_month(ban),\
                    image_pathlist=["1198/GRAYlspine", "1198/GRAYhip1"])
    set[0]
    print(len(set))
    split = spliter(set,rewrite_file=True)