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


def get_nsclc(args, alg, include_lb_to_ulb=True, splitNum=5, target_type="time"):
    crop_size = args.img_size
    crop_ratio = args.crop_ratio
    split_id = args.split_id
    dataset_path = args.data_dir
    ulb_rate = args.ulb_rate
    info_path = os.path.join(dataset_path,"NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")
    info_data = pd.read_csv(info_path)
    image_path = "manifest-1603198545583/NSCLC-Radiomics"
    get_meta = get_NSCLC_meta(num_classes=args.num_classes, target_type=args.target_type)
    # mean = [torch.tensor(get_meta(info_data,mean=True)['prompt'][i]) for i in range(len(get_meta(info_data,mean=True)['prompt']))]
    # std = [(torch.tensor(get_meta(info_data,std=True)['prompt'][i])) for i in range(len(get_meta(info_data,std=True)['prompt']))]
    mean = None
    std = None

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
    ])

    transform_medium = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 5),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        # RandAugment(3, 5),
        RandAugment(2, 5),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize([crop_size,crop_size]),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
    ])


    # 遍历所有图像,获取{patientID,images path}
    image_dict={} #所有patientID的MRIseries位置
    img_full_path = os.path.join(dataset_path, image_path)
    for patientID in os.listdir(img_full_path):
        if not patientID.startswith("LUNG1"):
            continue
        patientID_path = os.path.join(img_full_path, patientID)
        studyID_list=[]
        for studyID in os.listdir(patientID_path):
            studyID_list.append(studyID)
        if len(studyID_list)>1:
            print(patientID,studyID_list)
            continue
        elif len(studyID_list)<=0:
            print(patientID,"missing image")
            continue
        else:
            studyID_path = os.path.join(patientID_path,studyID_list[0])
            a_series=None
            for series in os.listdir(studyID_path):
                if(len(os.listdir(os.path.join(studyID_path,series))) > args.slides_num):
                    a_series = series
                    break
            if a_series is None:
                print(patientID,f"MRI numbers less than {args.slides_num}")
                continue
            series_path = os.path.join(studyID_path,a_series)
            image_dict[patientID]=series_path    

    print("DATASET: data init")
    filter_lb = filter_NSCLC_knowTime(ban)
    filter_ulb = filter_NSCLC_unknowTime(ban)
    filter_list = [filter_lb, filter_ulb]
    data_dict = [{'NO_list':[],'targets':[]} for _ in range(len(filter_list))] #targets好像没用上？忘了
    NO_list = image_dict.keys()
    for NO in NO_list:
        for i, filter in enumerate(filter_list):
            if filter.filtrate(NO, info_data):
                data_dict[i]['NO_list'].append(NO)
    lb_NO = data_dict[0]['NO_list']
    ulb_NO = data_dict[1]['NO_list']

    spliter_lb_ulb = spliter(lb_NO, ulb_NO, ulb_rate, ID="lb", splitNum=splitNum, path=args.save_dir)
    train_lb_NO, test_lb_NO, train_ulb_NO, test_ulb_NO = spliter_lb_ulb.get_split(split_id)
    if include_lb_to_ulb==True:
        train_ulb_NO = list(set(train_lb_NO) | set(train_ulb_NO))

    lb_dset = BasicDataset(
        NO_list=train_lb_NO,
        get_meta=get_meta,
        info_data=info_data,
        image_dict=image_dict,
        alg=alg,
        gray=False,
        transform=transform_weak,
        is_ulb=False,
        medium_transform=transform_strong,
        strong_transform=transform_strong,
        onehot=False,)
    ulb_dset = BasicDataset(
        NO_list=train_ulb_NO,
        get_meta=get_meta,
        info_data=info_data,
        image_dict=image_dict,
        alg=alg,
        gray=False,
        transform=transform_weak,
        is_ulb=True,
        medium_transform=transform_medium,
        strong_transform=transform_strong,
        onehot=False,)
    eval_dset = BasicDataset(
        NO_list=test_lb_NO,
        get_meta=get_meta,
        info_data=info_data,
        image_dict=image_dict,
        alg=alg,
        gray=False,
        transform=transform_val,
        is_ulb=False,
        medium_transform=None,
        strong_transform=None,
        onehot=False,)
    
    
    print("Dataset lb: {}".format(len(lb_dset)))
    print("Dataset ulb: {}".format(len(ulb_dset)))

    return lb_dset, ulb_dset, eval_dset, mean, std
        
class get_NSCLC_meta(object):
    def __init__(self, num_classes, target_type="time"):
        super().__init__()
        self.num_classes = num_classes
        self.target_type = target_type
    def __call__(self, df, NO=None, mean=False, std=False):
        if mean:
            return None
        elif std:
            return None
        else:
            row = df.loc[df['PatientID'].apply(str)==NO]
            if row.size <= 0:
                return None
            else:
                row = row.iloc[0]
        stageDict={"I":1,"II":2,"IIIa":3,"IIIb":4,"NA":2.5}
        histologyDict={"adenocarcinoma":1,"large cell":2,"nos":3,"squamous cell carcinoma":4,"NA":0}
        genderDict={"female":0,"male":1}

        age = row["age"]
        TStage = row["clinical.T.Stage"]
        Nstage = row["Clinical.N.Stage"]
        overallStage = row["Overall.Stage"]
        histology = row["Histology"]
        gender = row["gender"]
        time = row["Survival.time"]
        ifHappen = row["deadstatus.event"]

        if pd.isna(age):
            age = 68
        if pd.isna(TStage):
            TStage = 2.475
        if pd.isna(overallStage):
            overallStage = stageDict["NA"]
        else:
            overallStage = stageDict[overallStage]
        if pd.isna(histology):
            histology = histologyDict["NA"]
        else:
            histology = histologyDict[histology]
        gender = genderDict[gender]
        time = math.ceil(time/30)
        prompt = [age,TStage,Nstage,overallStage,histology,gender]

        if self.target_type=="ifHappen":
            target = ifHappen
        elif self.target_type=="time":
            target = time
        elif not (mean or std):
            raise RuntimeError("unknow target type in fracture dataset")
        else:
            target=None
        return {"prompt":[prompt], "target":target}


class filter_NSCLC_knowTime(object):
    def __init__(self, ban=[]):
        self.ban = ban
    def filtrate(self, NO, df):
        """
        return True when item should be reserve
        """    
        row = df.loc[df['PatientID'].apply(str)==NO]
        if len(row)==0:
            #print("miss info: ",NO)
            return False
        if NO not in self.ban and row['deadstatus.event'].values[0]==1:
            return True
        else:
            return False

class filter_NSCLC_unknowTime(object):
    def __init__(self, ban=[]):
        self.ban = ban
    def filtrate(self, NO, df):
        """
        return True when item should be reserve
        """    
        row = df.loc[df['PatientID'].apply(str)==NO]
        if len(row)==0:
            #print("miss info: ",NO)
            return False
        if NO not in self.ban and row['deadstatus.event'].values[0]==0:
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
        row = df.loc[df['PatientID'].apply(str)==NO]
        if len(row)==0:
            #print("miss info: ",NO)
            return False
        if NO not in self.ban:
            return True
        else:
            return False


class spliter:
    def __init__(self, NO_list_lb, NO_list_ulb, ulb_rate, ID="Default", path="./", shuffle=True, rewrite_file=False, splitNum=5):
        """
        :full_dataset 
        :filename save the split result
        :shuffle shuffle when it is True
        :update_file if cover with a new split.json
        :splitNum the ratio of testset
        """
        filename = f"split_Dataset-{ID}-{ulb_rate}_splitNum-{splitNum}.json"
        filepath = os.path.join(path, filename)
        print(filepath,os.path.exists(filepath))
        self.splitNum = splitNum
        if os.path.exists(filepath) and not rewrite_file:
            with open(filepath, 'r') as f:
                self.splitdict = json.load(f)
        else:
            dataset_size = len(NO_list_lb)+len(NO_list_ulb)
            lb_num = len(NO_list_lb)
            ulb_num = len(NO_list_ulb)
            if ulb_num/dataset_size < ulb_rate:
                lb2ulb_num = int((dataset_size)*ulb_rate - ulb_num)
                lb_num -= lb2ulb_num
            else:
                lb2ulb_num = 0
            f = open(filepath, 'w')
            self.splitdict={"lb":[],"lb2ulb":[],"ulb":[]}
            if shuffle:
                numpy.random.seed(numpy.int64(time.time()))
                numpy.random.shuffle(NO_list_lb)
                numpy.random.shuffle(NO_list_ulb)
            for i in range(splitNum):
                self.splitdict["lb"].append(NO_list_lb[ i*(lb_num//splitNum) : (i+1)*(lb_num//splitNum) ])
                self.splitdict["lb2ulb"].append(NO_list_lb[ lb_num+i*(lb2ulb_num//splitNum) : lb_num+(i+1)*(lb2ulb_num//splitNum) ])
                self.splitdict["ulb"].append(NO_list_ulb[ i*(ulb_num//splitNum) : (i+1)*(ulb_num//splitNum) ])
            json.dump(self.splitdict, f)


    def get_split(self, it=-1):
        train_lb_NO_list = []
        test_lb_NO_list = []
        train_ulb_NO_list = []
        test_ulb_NO_list = []
        for i in range(self.splitNum):
            if i!=it:
                train_lb_NO_list += self.splitdict["lb"][i]
                train_ulb_NO_list += self.splitdict["lb2ulb"][i]+self.splitdict["ulb"][i]
            if i==it:
                test_lb_NO_list += self.splitdict["lb"][i]+self.splitdict["lb2ulb"][i]
                test_ulb_NO_list += self.splitdict["ulb"][i]
        return train_lb_NO_list,test_lb_NO_list,train_ulb_NO_list,test_ulb_NO_list


if __name__ == "__main__":
    spliter_lb_ulb = spliter([], [], 0.8, ID="lb", splitNum=5, path="./saved_models/usb_ns_mini")
    train_lb_NO, test_lb_NO, train_ulb_NO, test_ulb_NO = spliter_lb_ulb.get_split(0)
    print(train_lb_NO, test_lb_NO, train_ulb_NO, test_ulb_NO)