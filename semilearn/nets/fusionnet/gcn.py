import torch
import math
from torch import nn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import numpy as np
import scipy.sparse as sp


class GraphConvolution(Module):                            
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, batch_input, adj):             # 这里代码做了简化,省去拉普拉斯算子。
        if batch_input.dim()==2:
            batch_input=torch.unsqueeze(batch_input,dim=0)
        output_list = []
        for input in batch_input:
            support = torch.mm(input, self.weight) # (nodes, nclass) = (nodes, nfeat) X (nfeat, nclass)
            output = torch.spmm(adj, support)      # (nodes, nclass) = (nodes, nodes) X (nodes, nclass)
            if self.bias is not None:
                output_list.append(output + self.bias)          # 加上偏置 (nodes, nclass)
            else:
                output_list.append(output)                      # (nodes, nclass)
        return torch.stack(output_list, dim=0)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):                                             # 定义两层GCN
    def __init__(self, nfeat:int, nclass:int, nhid:list, dropout:float):
        super(GCN, self).__init__()

        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nclass)
        self.nfeat = nfeat
        self.layers = len(nhid)+1
        self.size_list = [nfeat]+nhid+[nclass]
        self.gc = nn.ModuleList([
            GraphConvolution(self.size_list[i], self.size_list[i+1])
            for i in range(self.layers)
        ])
        self.dropout = dropout

    def forward(self, x, adj):
        # x = torch.nn.functional.relu(self.gc1(x, adj))
        # x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        if x.dim()<=2: #B,nodes*nfeat
            x = x.reshape(x.shape[0], -1, self.nfeat)
        x = x.float()
        x = self.gc[0](x, adj)
        for i in range(1,self.layers):
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
            x = self.gc[i](x, adj)
        return torch.nn.functional.log_softmax(x, dim=1)       # 对每一个节点做softmax


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_adj(path, node_num=52):
    edges_unordered = np.genfromtxt(path, dtype=np.int32)     # 读取边信息
    #edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    edges = edges_unordered
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(node_num, node_num),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)   
    adj_nm = normalize(adj + sp.eye(adj.shape[0]))
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj_nm)
    return adj_tensor


#代码解释 https://zhuanlan.zhihu.com/p/390836594
if __name__ == "__main__":
    import os
    from torch import optim
    print(os.getcwd())
    B=2
    node_num=10
    node_feat=1

    adj = load_adj("adjTest.cites",node_num=node_num)
    print(adj)
    x = torch.rand(B,node_num,node_feat)
    net = GCN(nfeat=1,nclass=2,nhid=[2,4],dropout=0.5)
    net.train()
    opt = optim.Adam(net.parameters(), lr=0.005)
    output = net(x,adj)
    print(output)