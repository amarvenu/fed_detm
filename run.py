import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import scipy

import torch
from torch import nn, optim
from torch.nn import functional as F

from src import data
from src.detm import DETM

from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

from analysis_utils import get_detm_topics, topic_diversity


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
def get_args(emb_dim):
    ### data and file related arguments
    arg_str = """
    parser.add_argument('--dataset', type=str, default='un', help='name of corpus')
    parser.add_argument('--data_path', type=str, default='un/', help='directory containing data')
    parser.add_argument('--emb_path', type=str, default='skipgram/embeddings.txt', help='directory containing embeddings')
    parser.add_argument('--save_path', type=str, default='./results', help='path to save results')
    parser.add_argument('--batch_size', type=int, default=1000, help='number of documents in a batch for training')
    parser.add_argument('--min_df', type=int, default=100, help='to get the right data..minimum document frequency')

    ### model-related arguments
    parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
    parser.add_argument('--rho_size', type=int, default=300, help='dimension of rho')
    parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
    parser.add_argument('--t_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
    parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
    parser.add_argument('--train_embeddings', type=int, default=1, help='whether to fix rho or train it')
    parser.add_argument('--eta_nlayers', type=int, default=3, help='number of layers for eta')
    parser.add_argument('--eta_hidden_size', type=int, default=200, help='number of hidden units for rnn')
    parser.add_argument('--delta', type=float, default=0.005, help='prior variance')

    ### optimization-related arguments
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--mode', type=str, default='train', help='train or eval model')
    parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
    parser.add_argument('--seed', type=int, default=2019, help='random seed (default: 1)')
    parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')
    parser.add_argument('--eta_dropout', type=float, default=0.0, help='dropout rate on rnn for eta')
    parser.add_argument('--clip', type=float, default=0.0, help='gradient clipping')
    parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
    parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
    parser.add_argument('--anneal_lr', type=int, default=0, help='whether to anneal the learning rate or not')
    parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

    ### evaluation, visualization, and logging-related arguments
    parser.add_argument('--num_words', type=int, default=20, help='number of words for topic viz')
    parser.add_argument('--log_interval', type=int, default=10, help='when to log training')
    parser.add_argument('--visualize_every', type=int, default=1, help='when to visualize results')
    parser.add_argument('--eval_batch_size', type=int, default=1000, help='input batch size for evaluation')
    parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
    parser.add_argument('--tc', type=int, default=0, help='whether to compute tc or not')
    """.split('\n')

    keys = [x.strip("parser.add_argument('").split(',')[0].strip('--').strip("'") for x in arg_str if (len(x) > 0) and (not x.startswith('#'))]
    values = [x.strip("parser.add_argument('").split(',')[2].strip(" default=").strip("'") for x in arg_str if (len(x) > 0) and (not x.startswith('#'))]
    tmp_dict = dict(zip(keys, values))

    for k, v in tmp_dict.items():
        if v.isnumeric():
            tmp_dict[k] = int(v)
        elif ('.' in v) and (v[0].isnumeric()):
            tmp_dict[k] = float(v)    

    args = AttrDict()
    args.update(tmp_dict)

    args.train_embeddings = 0
    args.rho_size = emb_dim
    args.num_topics = 10
    args.batch_size = 100
    args.num_words = 10
    
    return args


def process_data(file: str, args):
    train_arr = np.load(file, allow_pickle=True)

    train_tokens = train_arr['train_tokens']
    train_counts = train_arr['train_counts']
    train_times = train_arr['train_times']
    vocab = train_arr['vocab']
    embeddings = train_arr['embeddings']

    args.num_times = len(np.unique(train_times))
    args.num_docs_train = len(train_tokens)
    args.vocab_size = len(vocab)
    
    train_rnn_inp = data.get_rnn_input(train_tokens, train_counts, train_times, args.num_times, args.vocab_size, args.num_docs_train)

    return train_rnn_inp, train_tokens, train_counts, train_times, vocab, embeddings, args


def prep_files(args, prefix):
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
        
    if args.mode == 'eval':
        ckpt = args.load_from
    else:
        ckpt = os.path.join(args.save_path, 
            prefix+'_detm_{}_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_L_{}_minDF_{}_trainEmbeddings_{}'.format(
            args.dataset, args.num_topics, args.t_hidden_size, args.optimizer, args.clip, args.theta_act, 
                args.lr, args.batch_size, args.rho_size, args.eta_nlayers, args.min_df, args.train_embeddings))
    return ckpt
def get_model(args, embeddings):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embeddings = torch.from_numpy(embeddings).to(device)
    args.embeddings_dim = embeddings.size()

    model = DETM(args, embeddings)
    model.to(device)
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    elif args.optimizer == 'asgd':
        optimizer = optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
    else:
        print('Defaulting to vanilla SGD')
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    return model, optimizer, args


def train(epoch, model, optimizer, train_tokens, train_counts, train_times, train_rnn_inp, args):
    """Train DETM on data for one epoch.
    """
    model.train()
    acc_loss = 0
    acc_nll = 0
    acc_kl_theta_loss = 0
    acc_kl_eta_loss = 0
    acc_kl_alpha_loss = 0
    cnt = 0
    indices = torch.randperm(args.num_docs_train)
    indices = torch.split(indices, args.batch_size) 
    for idx, ind in enumerate(indices):
        optimizer.zero_grad()
        model.zero_grad()
        data_batch, times_batch = data.get_batch(
            train_tokens, train_counts, ind, args.vocab_size, args.emb_size, temporal=True, times=train_times)
        sums = data_batch.sum(1).unsqueeze(1)
        if args.bow_norm:
            normalized_data_batch = data_batch / sums
        else:
            normalized_data_batch = data_batch

        loss, nll, kl_alpha, kl_eta, kl_theta = model(data_batch, normalized_data_batch, times_batch, train_rnn_inp, args.num_docs_train)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        acc_loss += torch.sum(loss).item()
        acc_nll += torch.sum(nll).item()
        acc_kl_theta_loss += torch.sum(kl_theta).item()
        acc_kl_eta_loss += torch.sum(kl_eta).item()
        acc_kl_alpha_loss += torch.sum(kl_alpha).item()
        cnt += 1

        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = round(acc_loss / cnt, 2) 
            cur_nll = round(acc_nll / cnt, 2) 
            cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
            cur_kl_eta = round(acc_kl_eta_loss / cnt, 2) 
            cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2) 
            lr = optimizer.param_groups[0]['lr']
            print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. KL_eta: {} .. KL_alpha: {} .. Rec_loss: {} .. NELBO: {}'.format(
                epoch, idx, len(indices), lr, cur_kl_theta, cur_kl_eta, cur_kl_alpha, cur_nll, cur_loss))
    
    cur_loss = round(acc_loss / cnt, 2) 
    cur_nll = round(acc_nll / cnt, 2) 
    cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
    cur_kl_eta = round(acc_kl_eta_loss / cnt, 2) 
    cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2) 
    lr = optimizer.param_groups[0]['lr']
    print('*'*100)
    print('Epoch----->{} .. LR: {} .. KL_theta: {} .. KL_eta: {} .. KL_alpha: {} .. Rec_loss: {} .. NELBO: {}'.format(
            epoch, lr, cur_kl_theta, cur_kl_eta, cur_kl_alpha, cur_nll, cur_loss))
    print('*'*100)