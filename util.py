import torch


def cindex_pair_generate(ifMOF):
    flag_index=[]
    for i,flag in enumerate(ifMOF):
        if flag==1 or flag=="1":
            flag_index.append(i)
    pair_index=[]
    for flag_a in flag_index:
        for flag_b in list(reversed(flag_index)):
            if flag_b==flag_a:
                break
            pair_index.append((flag_a, flag_b))
    return pair_index


def cindex(score, gt):
    assert len(score)>1, "utils: C-index need batchsize>1 !!!"
    pair_list = []
    indexs = range(len(score))
    for i in indexs:
        for j in list(reversed(indexs)):
            if i==j:
                break
            else:
                pair_list.append((i,j))
    loss_result = 0
    for pair in pair_list:
        if gt[pair[0]]-gt[pair[1]] == 0:
            if score[pair[0]] == score[pair[1]]:
                loss = 1
            else:
                loss = 0
        elif (score[pair[0]]-score[pair[1]])*(gt[pair[0]]-gt[pair[1]]) < 0:
            loss = 0
        else:
            loss = 1
        loss_result += loss
    return loss_result/len(pair_list)

def scoreCindexLoss(score_set, gt, gt_fracTime=0, gt_ifMOF=1):
    assert score_set.size(0)>1, "utils: C-index need batchsize>1 !!!"
    target = gt.transpose(0,1)[gt_fracTime] #fracTime
    ifMOF = gt.transpose(0,1)[gt_ifMOF] #ifMOF
    pair_list = cindex_pair_generate(ifMOF)
    result_list = []
    for score in score_set.transpose(0,1):
        loss_result = 0
        for pair in pair_list:
            if target[pair[0]]-target[pair[1]] == 0:
                if score[pair[0]] == score[pair[1]]:
                    loss = 1
                else:
                    loss = 0
            elif (score[pair[0]]-score[pair[1]])*(target[pair[0]]-target[pair[1]]) < 0:
                loss = 0
            else:
                loss = 1
            loss_result += loss
        result_list.append(loss_result/len(pair_list))
    return result_list

class CindexLoss(torch.nn.Module):
    def __init__(self,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        super(CindexLoss, self).__init__()
        self.device = device
    def forward(self, pred, gt, gt_fracTime=0, gt_ifMOF=1):
        assert pred.size(0)>1, "C-index need batchsize>1 !!!"
        target = gt.transpose(0,1)[gt_fracTime] #fracTime
        ifMOF = gt.transpose(0,1)[gt_ifMOF] #ifMOF
        pair_list = cindex_pair_generate(ifMOF)
        loss_result = 0
        for pair in pair_list:
            loss = (pred[pair[0]]-pred[pair[1]])*(target[pair[0]]-target[pair[1]])/100
            max_tuply = torch.max( loss, torch.tensor([0]).to(self.device) )
            loss_result += max_tuply[0]
            if max_tuply[0] < 0:
                print("?")
        return loss_result/len(pair_list)


if __name__=="__main__":
    device = torch.device("cuda")
    import numpy as np
    # ifMOF = torch.tensor([0,1,1,0,1,0,0,0,0,1])
    # print(cindex_pair_generate(ifMOF))
    
    # FRAXs = torch.tensor([[2,4,1,5,3,5,6],[5,2,5,2,4,3,5],[6,2,34,6,7,2,3],[3,6,376,4,2,5,3]]).to(device)
    # gt = torch.tensor([[4,1],[5,1],[7,1],[1,1]]).to(device)
    # result_list = scoreCindexLoss(score_set=FRAXs,gt=gt)
    # print(result_list)

    score = np.array([2,5,6,3])
    gt = np.array([4,5,7,1])
    result = cindex(score=score,gt=gt)
    print(result)