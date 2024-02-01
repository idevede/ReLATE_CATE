import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']="3"
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim, autograd
DATA_DIR="./"

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

def wasserstein(x,y,p=0.5,lam=10,its=10,sq=False,backpropT=False,cuda=True):
    """return W dist between x and y"""
    '''distance matrix M'''
    nx = x.shape[0]
    ny = y.shape[0]
    
    x = x.squeeze()
    y = y.squeeze()
    
#    pdist = torch.nn.PairwiseDistance(p=2)

    M = pdist(x,y) #distance_matrix(x,y,p=2)
    
    '''estimate lambda and delta'''
    M_mean = torch.mean(M)
    M_drop = F.dropout(M,10.0/(nx*ny))
    delta = torch.max(M_drop).detach()
    eff_lam = (lam/M_mean).detach()

    '''compute new distance matrix'''
    Mt = M
    row = delta*torch.ones(M[0:1,:].shape).cuda()
    col = torch.cat([delta*torch.ones(M[:,0:1].shape).cuda(),torch.zeros((1,1)).cuda()],0)
    if cuda:
        row = row.cuda()
        col = col.cuda()
        M = M.cuda()
    Mt = torch.cat([M,row],0)
    Mt = torch.cat([Mt,col],1)

    '''compute marginal'''
    a = torch.cat([p*torch.ones((nx,1))/nx,(1-p)*torch.ones((1,1))],0)
    b = torch.cat([(1-p)*torch.ones((ny,1))/ny, p*torch.ones((1,1))],0)

    '''compute kernel'''
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1)*1e-6
    if cuda:
        temp_term = temp_term.cuda()
        a = a.cuda()
        b = b.cuda()
    K = torch.exp(-Mlam).cuda() + temp_term.cuda()
    U = K * Mt
    ainvK = K.cuda()/a.cuda()

    u = a

    for i in range(its):
        u = 1.0/(ainvK.matmul(b.cuda()/torch.t(torch.t(u.cuda()).matmul(K.cuda()))))
        if cuda:
            u = u.cuda()
    v = b.cuda()/(torch.t(torch.t(u.cuda()).matmul(K.cuda())))
    if cuda:
        v = v.cuda()

    upper_t = u*(torch.t(v)*K).detach()

    E = upper_t*Mt
    D = 2*torch.sum(E)

    if cuda:
        D = D.cuda()

    return D, Mlam



parser = argparse.ArgumentParser(description='SpeedDATING')
parser.add_argument('--hidden_dim', type=int, default=250)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.0001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=1)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=100.0)
parser.add_argument('--steps', type=int, default=600)
parser.add_argument('--env', type=int, default=22)
parser.add_argument('--collider', type=int, default=1)
parser.add_argument('--num_col', type=int, default=0)
parser.add_argument('--mod', type=str, default='Mod1')
parser.add_argument('--dimension', type=str, default='low')
parser.add_argument('--dat', type=int, default=1)
parser.add_argument('--net', type=str, default='tarnet')
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--icp', type=int, default=1)
parser.add_argument('--all_train', type=int, default=0)
parser.add_argument('--data_base_dir', type=str, default="./")
parser.add_argument('--output_base_dir', type=str, default='./')
parser.add_argument('--reg', type=str, default='mine')
parser.add_argument('--reg_weight', type=float, default=0.5)
parser.add_argument('--p_step', type=int, default=500)

flags = parser.parse_args()

print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

final_train_accs = []
final_test_accs = []

final_train_ate = []
final_test_ate = []
final_train_treatment_acc=[]
final_test_treatment_acc=[]

torch.manual_seed(1)

# Load data
file_path = flags.data_base_dir
df_path = os.path.join(file_path,
                       '{}/speedDate{}{}{}.csv'.format(flags.mod, flags.mod, flags.dimension, str(flags.dat)))
df = pd.read_csv(df_path)
data = df.values
oracle = pd.read_csv(
    file_path + '{}/speedDate{}{}Oracle{}.csv'.format(flags.mod, flags.mod, flags.dimension, str(flags.dat)))
ITE_oracle = oracle['ITE'].values.reshape(-1, 1)

reader = pd.read_csv(file_path + flags.mod + '/ate_truth.csv')
truth = {
    'low': reader['low_ate'].values,
    'med': reader['med_ate'].values,
    'high': reader['high_ate'].values
}

Y = data[:, 0].reshape(-1, 1)
T = data[:, 1].reshape(-1, 1)
X = data[:, 2:]
index = X[:, flags.env].argsort()
X = np.delete(X, flags.env, axis=1)

share = int(index.shape[0] / 3)
env1 = index[:share]
env2 = index[share:share * 2]
env3 = index[share * 2: share * 3]


# Build environments
def make_environment(X, Y, T, ITE_oracle, env, e):
    if flags.collider == 1:
        X = X[env]
        for i in range(flags.num_col):
            np.random.seed(i)
            col_1 = Y[env] + T[env] + np.random.normal(0, e, len(env)).reshape(-1, 1)
            X[:, -(i + 1)] = np.squeeze(col_1)
    else:
        X = X[env]
#for runing with gpu & cpu
    if flags.gpu == 1:
        return {
            'covariates': torch.from_numpy(X).cuda(),
            'outcome': torch.from_numpy(Y[env]).cuda(),
            'treatment': torch.from_numpy(T[env]).cuda(),
            'ITE_oracle': torch.from_numpy(ITE_oracle[env])
        }
    else:
        return {
            'covariates': torch.from_numpy(X),
            'outcome': torch.from_numpy(Y[env]),
            'treatment': torch.from_numpy(T[env]),
            'ITE_oracle': torch.from_numpy(ITE_oracle[env])
        }


envs = [
    make_environment(X, Y, T, ITE_oracle,  env1, .01),
    make_environment(X, Y, T, ITE_oracle,  env2, .2),
    make_environment(X, Y, T, ITE_oracle, env3, 1),
]


#  # Define and instantiate the model
 # # Define and instantiate the model
class MLP(nn.Module):
    def __init__(self, dim):
        super(MLP, self).__init__()
        hidden_dim = flags.hidden_dim
        hypo_dim = 100
        self.lin1 = nn.Linear(dim, hidden_dim)
        self.lin1_1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin1_2 = nn.Linear(hidden_dim, hypo_dim)
        # tarnet
        if flags.net == 'tarnet':
            self.lin1_3 = nn.Linear(dim, 1)
        else:
            self.lin1_3 = nn.Linear(hypo_dim, 1)

        self.lin_n = nn.Linear(hypo_dim, hypo_dim)
        self.lin_o = nn.Linear(hypo_dim, hypo_dim)

        self.lin2_0 = nn.Linear(hypo_dim, hypo_dim)
        self.lin2_1 = nn.Linear(hypo_dim, hypo_dim)

        self.lin3_0 = nn.Linear(hypo_dim, hypo_dim)
        self.lin3_1 = nn.Linear(hypo_dim, hypo_dim)

        self.lin4_0 = nn.Linear(hypo_dim, 1)
        self.lin4_1 = nn.Linear(hypo_dim, 1)

        self.lin2_0_n = nn.Linear(hypo_dim, hypo_dim)
        self.lin2_1_n = nn.Linear(hypo_dim, hypo_dim)

        self.lin3_0_n = nn.Linear(hypo_dim, hypo_dim)
        self.lin3_1_n = nn.Linear(hypo_dim, hypo_dim)

        self.lin4_0_n = nn.Linear(hypo_dim, 1)
        self.lin4_1_n = nn.Linear(hypo_dim, 1)

        if flags.gpu ==1:
            self.a=torch.nn.Parameter(torch.tensor(1.)).cuda()#.requires_grad_()
            self.b=torch.nn.Parameter(torch.tensor(1.)).cuda()#.requires_grad_()

        else:
            self.a = torch.nn.Parameter(torch.tensor(1.))#.requires_grad_()
            self.b = torch.nn.Parameter(torch.tensor(1.))#.requires_grad_()



        self.lin_mi_0 = nn.Linear(2 * hypo_dim, hypo_dim)
        self.lin_mi_1 = nn.Linear(hypo_dim, 1)

        for lin in [self.lin1, self.lin1_1, self.lin1_2, self.lin2_0, self.lin2_1, self.lin1_3, self.lin3_0,
                    self.lin3_1, self.lin4_0, self.lin4_1,  self.lin_mi_0, self.lin_mi_1, self.lin2_0_n,
                    self.lin2_1_n, self.lin3_0_n, self.lin3_1_n, self.lin4_0_n, self.lin4_1_n, self.lin_n,
                    self.lin_o]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, input):
        initial = input.view(input.shape)

        if flags.net != 'tarnet':
            ho_kl_d = F.relu(self.lin_o(initial)) # xo
            ht_kl_d = F.relu(self.lin_t(initial)) # xo
            hn_kl_d = F.relu(self.lin_n(initial)) # xn
        x = F.relu(self.lin1(initial))
        x = F.relu(self.lin1_1(x))
        x = F.relu(self.lin1_2(x))
        x = F.relu(x)
        

        hn_kl = F.relu(self.lin_n(x)) # xn
        ho_kl = F.relu(self.lin_o(x)) # xo
        ht_kl = F.relu(self.lin_t(x))

        loss = torch.tensor(0.)
        if flags.reg == 'mine':

            ######MINE
            batch_size = hn_kl.size(0)
            tiled_x = torch.cat([hn_kl, hn_kl, ], dim=0)
            idx = torch.randperm(batch_size)

            shuffled_y = ho_kl[idx]
            concat_y = torch.cat([ho_kl, shuffled_y], dim=0)
            inputs = torch.cat([tiled_x, concat_y], dim=1)
            logits = F.relu(self.lin_mi_0(inputs))
            # logits = F.relu(self.lin_mi_1(logits))
            logits = self.lin_mi_1(logits)

            pred_xy = logits[:batch_size]
            pred_x_y = logits[batch_size:]
            loss = np.log2(np.exp(1)) * torch.abs((torch.log(torch.mean(torch.exp(pred_x_y)))-torch.mean(pred_xy)))
            # x_n, x_t

            batch_size = hn_kl.size(0)
            tiled_x = torch.cat([hn_kl, hn_kl, ], dim=0)
            idx = torch.randperm(batch_size)
            # import ipdb; ipdb.set_trace()
            shuffled_y = ht_kl[idx]
            concat_y = torch.cat([ht_kl, shuffled_y], dim=0)
            inputs = torch.cat([tiled_x, concat_y], dim=1)
            logits = F.relu(self.lin_mi_t_0(inputs))
            # logits = F.relu(self.lin_mi_1(logits))
            logits = self.lin_mi_t_1(logits)

            pred_xy = logits[:batch_size]
            pred_x_y = logits[batch_size:]
            loss += np.log2(np.exp(1)) * torch.abs((torch.log(torch.mean(torch.exp(pred_x_y))) - torch.mean(pred_xy)))
    
            #loss = -loss
            # # compute loss, you'd better scale exp to bit
            # ######Done MINE
        elif flags.reg == 'wasserstein':
            ###wasserstein
            loss, Mlam = -1 * wasserstein(hn_kl, ho_kl, cuda = False)
            ###
        if flags.net == 'tarnet':
            t = self.lin1_3(hn_kl + ht_kl)
        else:
            t = self.lin1_3(hn_kl_d + ht_kl_d)

        # h1, h2 - different group
        # xn h1_kl
        h_0_n = F.relu(self.lin2_0_n(hn_kl)) # xn
        h_1_n = F.relu(self.lin2_1_n(hn_kl)) # xn

        h_1_n = F.relu(h_1_n)
        h_0_n = F.relu(h_0_n)

        h0_p_n = F.relu(self.lin3_0_n(h_0_n))
        h1_p_n = F.relu(self.lin3_1_n(h_1_n))

        h0_n = self.lin4_0_n(h0_p_n)
        h1_n = self.lin4_1_n(h1_p_n)


        # h1, h2 - different group
        # xn h1_kl
        h_0 = F.relu(self.lin2_0(ho_kl)) # xo
        h_1 = F.relu(self.lin2_1(ho_kl)) # xo

        h_1 = F.relu(h_1)
        h_0 = F.relu(h_0)

        h0_p = F.relu(self.lin3_0(h_0))
        h1_p = F.relu(self.lin3_1(h_1))

        

        h0 = self.lin4_0(h0_p)
        h1 = self.lin4_1(h1_p)

        h0_p_a =torch.add(h0_p, h0_p_n)
        h1_p_a =torch.add(h0_p, h1_p_n)

        # h0_f = torch.add(self.a*h0, self.b*h0_n)
        # h1_f = torch.add(self.a*h0, self.b*h1_n)
        h0_f = torch.add(h0, h0_n)
        h1_f = torch.add(h0, h1_n)


        return torch.cat((h0_f, h1_f, t), 1), h0_p_a, h1_p_a, loss #, h0, h1
        #return torch.cat((h0+h0_n, h1+h1_n, t), 1), h0_p_n+h0_p, h1_p_n+h1_p, loss
        #return torch.cat((h0+h0_n, h1+h1_n, t), 1), h0_p_n, h1_p_n, loss, h0, h1

mlp = MLP(X.shape[1])
if flags.gpu==1:
    mlp.cuda()
else:
    mlp


# Define loss function helpers
def mean_nll(y_logit, y):
    return nn.functional.binary_cross_entropy_with_logits(y_logit, y.float())

def mean_accuracy(y_logit, y):
    preds = (y_logit > 0.).double()
    return ((preds - y).abs() < 1e-2).float().mean()


def penalty(y_logit, y):
    if flags.gpu ==1:
        scale=torch.tensor(1.).cuda().requires_grad_()
    else:
        scale = torch.tensor(1.).requires_grad_()
    loss = mean_nll(y_logit * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    res = torch.sum(grad ** 2)
    return res

def penalty_coco(y_logit, y):
    if flags.gpu ==1:
       
        scale = torch.nn.Parameter(torch.normal(1,0.2,[y_logit.shape[1], 1])).cuda().requires_grad_()
    else:
        scale = torch.nn.Parameter(torch.normal(1,0.2,[y_logit.shape[1], 1])).requires_grad_()

    loss = mean_nll(y_logit @ scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    res = torch.abs(torch.mean((grad**4 *scale)))

    return res



def ite(y0_logit, y1_logit):
    y0_pred = torch.sigmoid(y0_logit).float()
    y1_pred = torch.sigmoid(y1_logit).float()
    return y1_pred - y0_pred


def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))


pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc', 'train_ate', 'test_ate', 'train_t_acc',
             'test_t_acc')


def train_stack(val, env, all_train=True):
    if all_train == 1:
        return torch.stack([env[0][val], env[1][val], env[2][val]]).mean()
    else:
        return torch.stack([env[0][val], env[1][val]]).mean()


#different choice of optimizer yields different performance.
optimizer_adam = optim.Adam(mlp.parameters(), lr=flags.lr)
optimizer_sgd = optim.SGD(mlp.parameters(), lr=1e-7, momentum=0.9)

# training loop
for step in range(flags.steps):
    for env in envs:
        logits, h0_p, h1_p, mi_loss = mlp(env['covariates'].float())
        y0_logit = logits[:, 0].unsqueeze(1)
        y1_logit = logits[:, 1].unsqueeze(1)
        t_logit = logits[:, 2].unsqueeze(1)
        t = env['treatment'].float()
        y_logit = t * y1_logit + (1 - t) * y0_logit
        #yo_logit = t * y1 + (1 - t) * y0
        y_penalty =  t*h1_p-  (1-t)*h0_p

        env['ite'] = ite(y0_logit, y1_logit)

        env['nll'] = mean_nll(y_logit, env['outcome'])
        env['t_nll'] = mean_nll(t_logit, t)

        env['acc'] = mean_accuracy(y_logit, env['outcome'])
        env['t_acc'] = mean_accuracy(t_logit, env['treatment'])
        #env['penalty'] = penalty(y_logit, env['outcome'])
        env['penalty'] = 0.8* penalty_coco(y_penalty, env['outcome'])
        env['kl'] = mi_loss #kl_loss(h0_kl, h1_kl)

    all_train = flags.all_train
    train_nll = train_stack('nll', envs, all_train=all_train)
    train_t_nll = train_stack('t_nll', envs, all_train=all_train)
    train_acc = train_stack('acc', envs, all_train=all_train)
    train_t_acc = train_stack('t_acc', envs, all_train=all_train)
    train_kl = train_stack('kl', envs, all_train=all_train)


    train_ate = train_stack('ite', envs, all_train=all_train)
    train_penalty = train_stack('penalty', envs, all_train=all_train)
    if flags.gpu == 1:
        weight_norm = torch.tensor(0.).cuda()
    else:
        weight_norm = torch.tensor(0.)
    for w in mlp.parameters():
        weight_norm += w.norm().pow(2)

    loss = train_nll.clone()
    if not torch.isinf(train_kl).any() and flags.penalty_weight!=0:
    #if not torch.isinf(train_kl).any() and flags.penalty_weight!=0 and step< flags.p_step and not torch.isnan(train_kl).any():
        loss +=   flags.reg_weight* train_kl.clone() #+ train_t_nll.clone()

    loss += flags.l2_regularizer_weight * weight_norm
    penalty_weight = (flags.penalty_weight
                      if step >= flags.penalty_anneal_iters else 1)
    loss += penalty_weight * train_penalty
    loss += train_t_nll

    if penalty_weight > 1.0:
        # Rescale the entire loss to keep gradients in a reasonable range
        loss /= penalty_weight
    if step < 501:
        optimizer_adam.zero_grad()
        loss.backward()
        optimizer_adam.step()
    #train with sgd after the performance is stablized with adam.
    else:
        optimizer_sgd.zero_grad()
        loss.backward()
        optimizer_sgd.step()

    test_acc = envs[2]['acc']
    test_t_acc = envs[2]['t_acc']
    test_ate = envs[2]['ite'].mean()

    pred_ite = torch.stack([envs[0]['ite'], envs[1]['ite'], envs[2]['ite']]).detach().cpu().numpy()
    true_ite = torch.stack([envs[0]['ITE_oracle'], envs[1]['ITE_oracle'], envs[2]['ITE_oracle']]).detach().cpu().numpy()

    if step % 100 == 0:
        pretty_print(
            np.int32(step),
            train_nll.detach().cpu().numpy() + train_t_nll.detach().cpu().numpy(),
            train_acc.detach().cpu().numpy(),
            train_penalty.detach().cpu().numpy(),
            test_acc.detach().cpu().numpy(),
            np.mean(train_ate.detach().cpu().numpy()),
            np.mean(test_ate.detach().cpu().numpy()),
            train_t_acc.detach().cpu().numpy(),
            test_t_acc.detach().cpu().numpy()
        )
final_time = time.time()

# converting the tensor to numpy arrays
final_train_accs.append(train_acc.detach().cpu().numpy())
final_test_accs.append(test_acc.detach().cpu().numpy())
final_train_treatment_acc.append(train_t_acc.detach().cpu().numpy())
final_test_treatment_acc.append(test_t_acc.detach().cpu().numpy())

final_train_ate.append(train_ate.detach().cpu().numpy())
final_test_ate.append(test_ate.detach().cpu().numpy())

print('Final train acc (mean/std across restarts so far):')
print(np.mean(final_train_accs), np.std(final_train_accs))

print('Final test acc (mean/std across restarts so far):')
print(np.mean(final_test_accs), np.std(final_test_accs))

print('Final train ate mae is:')
print(abs(np.mean(final_train_ate) - truth[flags.dimension]))

print('Final test ate mae is:')
print(abs(np.mean(final_test_ate) - truth[flags.dimension]))

print("PEHE is {}, std is: {}".format(np.square(pred_ite - true_ite).mean(), abs(pred_ite - true_ite).std()))
print("MAE for ate is ", (abs((pred_ite).mean() - truth[flags.dimension])))

saver = {
    'pred_ite': [pred_ite],
    'sample_ite': [true_ite],
    'Y': torch.stack([envs[0]['outcome'], envs[1]['outcome'], envs[2]['outcome']]).detach().cpu().numpy(),
    'T': torch.stack([envs[0]['treatment'], envs[1]['treatment'], envs[2]['treatment']]).detach().cpu().numpy(),
    'index': [index]
}
if flags.net=='tarnet':
    tmp = os.path.join(flags.output_base_dir, 'tarnet/')
else:
    tmp = os.path.join(flags.output_base_dir, 'dragon/')

if flags.collider==1:
    tmp = os.path.join(tmp, 'collider/')
else:
    tmp = os.path.join(tmp, 'no_collider/')
if flags.all_train == 1:
    log_path = os.path.join(tmp, "all_train/")
else:
    log_path = os.path.join(tmp, "train_test/")

os.makedirs(log_path, exist_ok=True)

save_path = os.path.join(log_path, "{}/{}/".format(flags.mod, flags.dimension))
os.makedirs(save_path, exist_ok=True)

# if flags.penalty_weight > 0:
#     for num, output in enumerate([saver]):
#         np.savez_compressed(os.path.join(save_path, "irm_ite_output_{}".format(str(flags.dat))), **output)
# else:
#     for num, output in enumerate([saver]):
#         np.savez_compressed(os.path.join(save_path, "erm_ite_output_{}".format(str(flags.dat))), **output)

#

# if flags.penalty_weight > 0:
#     for num, output in enumerate([saver]):
#         np.savez_compressed(os.path.join(save_path, "xoxn_ite_output_{}_kl_{}_{}_{}".format(str(flags.dat),flags.reg, str(flags.reg_weight), str(flags.penalty_weight))), **output)
# else:
#     for num, output in enumerate([saver]):
#         np.savez_compressed(os.path.join(save_path, "erm_ite_output_{}".format(str(flags.dat))), **output)
final_output = pd.DataFrame({
    'train_acc': final_train_accs,
    'test_acc': final_test_accs,
    'train_ate': final_train_ate,
    'test_ate': final_test_ate,
    'train_treatment_acc': final_train_treatment_acc,
    'test_treatment_acc': final_test_treatment_acc,
    'PEHE': np.square(pred_ite - true_ite).mean(),
    'MAE': abs((pred_ite).mean() - truth[flags.dimension])
})


if flags.penalty_weight > 0:
    tmp = save_path+ 'xoxn_output_{}_kl_{}_{}_{}.csv'.format(str(flags.dat), flags.reg, str(flags.reg_weight), str(flags.penalty_weight))
else:
    tmp = save_path + 'erm_ate_output_{}.csv'.format( str(flags.dat))
print(tmp)
final_output.to_csv(tmp, index=False)
