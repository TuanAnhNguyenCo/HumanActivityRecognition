import torch
import torch.nn as nn
# import torch.nn.functional as F
import mediapipe as mp


def find_adjacency_matrix():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    adj = torch.zeros((21, 21))
    for connection in mp_hands.HAND_CONNECTIONS:
        adj[connection[0], connection[1]] = 1
        adj[connection[1], connection[0]] = 1
    return adj


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, num_vetex, device, dropout=0.5, bias=True):
        super(GraphConvolution, self).__init__()

        self.alpha = 1.
        self.device = device
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.randn(
            input_dim, output_dim)).to(device)
        if bias:
            self.bias = nn.Parameter(torch.randn(output_dim)).to(device)
        else:
            self.bias = None

        for w in [self.weight]:
            nn.init.xavier_normal_(w)

    def normalize(self, m):
        rowsum = torch.sum(m, 0)
        r_inv = torch.pow(rowsum, -0.5)
        r_mat_inv = torch.diag(r_inv).float()

        m_norm = torch.mm(r_mat_inv, m)
        m_norm = torch.mm(m_norm, r_mat_inv)

        return m_norm

    def forward(self, adj, x):

        x = self.dropout(x)

        # K-ordered Chebyshev polynomial
        adj_norm = self.normalize(adj)
        sqr_norm = self.normalize(torch.mm(adj, adj))
        m_norm = (self.alpha*adj_norm + (1.-self.alpha)
                  * sqr_norm).to(self.device)

        x_tmp = torch.einsum('abcd,de->abce', x, self.weight)
        x_out = torch.einsum('ij,abid->abjd', m_norm, x_tmp)
        if self.bias is not None:
            x_out += self.bias
        x_out = self.act(x_out)

        return x_out


class StandConvolution(nn.Module):
    def __init__(self, dims, num_classes, dropout, device):
        super(StandConvolution, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3,
                      stride=(1, 2), padding=1),
            nn.InstanceNorm2d(dims[1]),
            nn.ReLU(inplace=True),
            # nn.AvgPool2d(3, stride=2),
            nn.Conv2d(dims[1], dims[2], kernel_size=3,
                      stride=(1, 2), padding=1),
            nn.InstanceNorm2d(
                dims[2]),
            nn.ReLU(
                inplace=True),
            # nn.AvgPool2d(3, stride=2),
            nn.Conv2d(dims[2], dims[3], kernel_size=3,
                      stride=(1, 2), padding=1),
            nn.InstanceNorm2d(
                dims[3]),
            nn.ReLU(
                inplace=True),
            # nn.AvgPool2d(3, stride=2)
        ).to(device)

        self.fc = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.dropout(x.permute(0, 3, 1, 2))
        x_tmp = self.conv(x)

        # x_out = self.fc(x_tmp.view(x.size(0), -1)) # multimodal
        x_out = x_tmp.view(x.size(0), -1)

        return x_out


class GGCN(nn.Module):
    def __init__(self, adj, num_classes, gc_dims, sc_dims, device, dropout=0.2):
        super(GGCN, self).__init__()

        adj = adj + torch.eye(adj.size(0)).to(adj).detach()
        ident = torch.eye(adj.size(0)).to(adj)
        zeros = torch.zeros(adj.size(0), adj.size(1)).to(adj)
        self.adj = torch.cat([torch.cat([adj, adj, zeros], 1),
                              torch.cat([adj, adj, adj], 1),
                              torch.cat([zeros, adj, adj], 1)], 0).float()

        self.gcl = GraphConvolution(
            gc_dims[0], gc_dims[1], 21, dropout=dropout, device=device)
        self.conv = StandConvolution(
            sc_dims, num_classes, dropout=dropout, device=device)

    def forward(self, x):
        # x: 1,75,21,3 => 1,75,21,3

        x = torch.cat([x[:, :-2], x[:, 1:-1], x[:, 2:]], dim=2)
        multi_conv = self.gcl(self.adj, x)
        logit = self.conv(multi_conv)  # (1,62,63,9)
        return logit


# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True


# seed_everything(20)
