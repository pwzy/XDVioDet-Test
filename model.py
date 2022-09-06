
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from layers import GraphConvolution, SimilarityAdj, DistanceAdj


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.1)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        n_features = args.feature_size
        n_class = args.num_classes

        self.conv1d1 = nn.Conv1d(in_channels=n_features, out_channels=512, kernel_size=1, padding=0)
        self.conv1d2 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1, padding=0)
        self.conv1d3 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=5, padding=2)
        self.conv1d4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        # Graph Convolution
        self.gc1 = GraphConvolution(128, 32, residual=True)  # nn.Linear(128, 32)
        self.gc2 = GraphConvolution(32, 32, residual=True)
        self.gc3 = GraphConvolution(128, 32, residual=True)  # nn.Linear(128, 32)
        self.gc4 = GraphConvolution(32, 32, residual=True)
        self.gc5 = GraphConvolution(128, 32, residual=True)  # nn.Linear(128, 32)
        self.gc6 = GraphConvolution(32, 32, residual=True)
        self.simAdj = SimilarityAdj(n_features, 32)
        self.disAdj = DistanceAdj()

        self.classifier = nn.Linear(32*3, n_class)
        self.approximator = nn.Sequential(nn.Conv1d(128, 64, 1, padding=0), nn.ReLU(),
                                          nn.Conv1d(64, 32, 1, padding=0), nn.ReLU())
        self.conv1d_approximator = nn.Conv1d(32, 1, 5, padding=0)
        self.dropout = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)



    def forward(self, inputs, seq_len):

        import ipdb
        ipdb.set_trace()

        # inputs.shape: [128, 200, 1152]
        
        x = inputs.permute(0, 2, 1)  # for conv1d  x.shape:  [128, 1152, 200]
        x = self.relu(self.conv1d1(x)) # x.shape:  [128, 512, 200]
        x = self.dropout(x)
        x = self.relu(self.conv1d2(x)) # x.shape:  [128, 128, 200]
        x = self.dropout(x)

        logits = self.approximator(x)  # logits.shape: [128, 32, 200]
        logits = F.pad(logits, (4, 0)) # logits.shape: [128, 32, 204]
        logits = self.conv1d_approximator(logits) # logits.shape: [128, 1, 200]
        logits = logits.permute(0, 2, 1)  # logits.shape: [128, 200, 1] 
        x = x.permute(0, 2, 1)  # b*t*c  # x.shape: [128, 200, 128]

        ## gcn
        scoadj = self.sadj(logits.detach(), seq_len)  # scoadj.shape: [128, 200, 200]
        adj = self.adj(inputs, seq_len)  # adj.shape: [128, 200, 200]
        disadj = self.disAdj(x.shape[0], x.shape[1])  # disadj.shape:  [128, 200, 200]
        x1_h = self.relu(self.gc1(x, adj)) # x1_h.shape: [128, 200, 32]
        x1_h = self.dropout(x1_h)
        x2_h = self.relu(self.gc3(x, disadj))
        x2_h = self.dropout(x2_h)
        x3_h = self.relu(self.gc5(x, scoadj))
        x3_h = self.dropout(x3_h)
        x1 = self.relu(self.gc2(x1_h, adj))
        x1 = self.dropout(x1)
        x2 = self.relu(self.gc4(x2_h, disadj))
        x2 = self.dropout(x2)
        x3 = self.relu(self.gc6(x3_h, scoadj))
        x3 = self.dropout(x3)
        x = torch.cat((x1, x2, x3), 2)
        x = self.classifier(x)
        return x, logits

    def sadj(self, logits, seq_len):
        lens = logits.shape[1]
        soft = nn.Softmax(1)
        logits2 = self.sigmoid(logits).repeat(1, 1, lens)
        tmp = logits2.permute(0, 2, 1)
        adj = 1. - torch.abs(logits2 - tmp)
        self.sig = lambda x:1/(1+torch.exp(-((x-0.5))/0.1))
        adj = self.sig(adj)
        output = torch.zeros_like(adj)
        if seq_len is None:
            for i in range(logits.shape[0]):
                tmp = adj[i]
                adj2 = soft(tmp)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = adj[i, :seq_len[i], :seq_len[i]]
                adj2 = soft(tmp)
                output[i, :seq_len[i], :seq_len[i]] = adj2
        return output


    def adj(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0,2,1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0,2,1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output


if __name__ == "__main__":
    import option
    args = option.parser.parse_args()
    device = torch.device('cuda:0')

    model = Model(args).to(device)

    input = torch.randn(128, 200, 1152).to(device)
    # seq_len = torch.tensor([200, 103, 200, 200, 200,  39, 200, 123,  79, 135, 200, 102, 127, 119,
         # 94, 165,  78, 170,  76, 103, 162, 200, 125, 156, 103,  57, 200, 200,
        # 185, 129, 120, 200,  74, 200, 126,  60, 200, 200,  30, 200, 200, 200,
        # 137,  98, 200,  34,  81, 200, 200,  51, 200,  42, 136, 200,  22, 200,
         # 82,  96, 200, 200, 200, 200, 200, 165,  21, 108,  98, 103, 200, 200,
         # 36, 200,  97, 200,  81,  94,  76,  21,  27,  67,  81, 144,  51,  82,
        # 157, 200,  82, 200, 200,  45,  60, 200, 200,  76, 200, 200, 122, 200,
        # 200, 178,  81,  62, 200, 200,  32, 117,  85, 200, 199, 200, 200, 200,
        # 200,  77, 200, 200,  34,  55, 105,  26, 105, 200, 200, 200, 200,  78,
        # 124, 200])
    seq_len = torch.tensor([200] * 128)

    logits, logits2 = model(input, seq_len)

    print(logits)
