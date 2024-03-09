import configparser
import re

import numpy as np
import torch as th
from .utils.activation import act_dict
import warnings


class Config(object):
    def __init__(self, file_path, model, dataset, task, gpu):
        conf = configparser.ConfigParser()
        try:
            conf.read(file_path)
        except:
            print("failed!")
        # training dataset path
        self.seed = 0
        self.patience = 1
        self.max_epoch = 1
        self.task = task
        self.model = model
        self.dataset = dataset
        if isinstance(dataset, str):
            self.dataset_name = dataset
        else:
            self.dataset_name = self.dataset.name
        if isinstance(model, str):
            self.model_name = model
        else:
            self.model_name = type(self.model).__name__
        self.optimizer = 'Adam'
        # custom model
        if self.model_name == 'RGCN':
            self.lr = conf.getfloat("RGCN", "learning_rate")
            self.dropout = conf.getfloat("RGCN", "dropout")

            self.in_dim = conf.getint("RGCN", "in_dim")
            self.hidden_dim = conf.getint("RGCN", "hidden_dim")

            self.n_bases = conf.getint("RGCN", "n_bases")
            self.num_layers = conf.getint("RGCN", "num_layers")
            self.max_epoch = conf.getint("RGCN", "max_epoch")
            self.weight_decay = conf.getfloat("RGCN", "weight_decay")
            self.seed = conf.getint("RGCN", "seed")
            self.fanout = conf.getint("RGCN", "fanout")
            self.patience = conf.getint("RGCN", "patience")
            self.batch_size = conf.getint("RGCN", "batch_size")
            self.validation = conf.getboolean("RGCN", "validation")
            self.mini_batch_flag = conf.getboolean("RGCN", "mini_batch_flag")
            self.use_self_loop = conf.getboolean("RGCN", "use_self_loop")
            self.use_uva = conf.getboolean("RGCN", "use_uva")
            # self.filtered = conf.get("RGCN", "filtered")

        elif self.model_name == 'CompGCN':
            self.lr = conf.getfloat("CompGCN", "learning_rate")

            self.weight_decay = conf.getfloat("CompGCN", "weight_decay")
            self.dropout = conf.getfloat("CompGCN", "dropout")

            self.in_dim = conf.getint("CompGCN", "in_dim")
            self.hidden_dim = conf.getint("CompGCN", "hidden_dim")
            self.out_dim = conf.getint("CompGCN", "out_dim")
            self.num_layers = conf.getint("CompGCN", "num_layers")
            self.max_epoch = conf.getint("CompGCN", "max_epoch")
            self.seed = conf.getint("CompGCN", "seed")
            self.patience = conf.getint("CompGCN", "patience")

            self.comp_fn = conf.get("CompGCN", "comp_fn")
            self.mini_batch_flag = conf.getboolean("CompGCN", "mini_batch_flag")
            self.validation = conf.getboolean("CompGCN", "validation")
            self.fanout = conf.getint("CompGCN", "fanout")
            self.batch_size = conf.getint("CompGCN", "batch_size")
            pass

        elif self.model_name == 'HAN':
            self.lr = conf.getfloat("HAN", "learning_rate")
            self.weight_decay = conf.getfloat("HAN", "weight_decay")
            self.seed = conf.getint("HAN", "seed")
            self.dropout = conf.getfloat("HAN", "dropout")

            self.hidden_dim = conf.getint('HAN', 'hidden_dim')
            self.out_dim = conf.getint('HAN', 'out_dim')
            num_heads = conf.get('HAN', 'num_heads').split('-')
            self.num_heads = [int(i) for i in num_heads]
            self.patience = conf.getint('HAN', 'patience')
            self.max_epoch = conf.getint('HAN', 'max_epoch')
            self.mini_batch_flag = conf.getboolean("HAN", "mini_batch_flag")

        elif self.model_name == 'HGT':
            self.lr = conf.getfloat("HGT", "learning_rate")
            self.weight_decay = conf.getfloat("HGT", "weight_decay")
            self.seed = conf.getint("HGT", "seed")
            self.dropout = conf.getfloat("HGT", "dropout")

            self.batch_size = conf.getint("HGT", "batch_size")
            self.hidden_dim = conf.getint('HGT', 'hidden_dim')
            self.out_dim = conf.getint('HGT', 'out_dim')
            self.num_heads = conf.getint('HGT', 'num_heads')
            self.patience = conf.getint('HGT', 'patience')
            self.max_epoch = conf.getint('HGT', 'max_epoch')
            self.num_workers = conf.getint("HGT", "num_workers")
            self.mini_batch_flag = conf.getboolean("HGT", "mini_batch_flag")
            self.fanout = conf.getint("HGT", "fanout")
            self.norm = conf.getboolean("HGT", "norm")
            self.num_layers = conf.getint("HGT", "num_layers")
            self.num_heads = conf.getint("HGT", "num_heads")
            self.use_uva = conf.getboolean("HGT", "use_uva")

        elif self.model_name == 'HPN':
            self.lr = conf.getfloat("HPN", "learning_rate")
            self.weight_decay = conf.getfloat("HPN", "weight_decay")
            self.seed = conf.getint("HPN", "seed")
            self.dropout = conf.getfloat("HPN", "dropout")
            self.hidden_dim = conf.getint('HPN', 'hidden_dim')
            self.k_layer = conf.getint("HPN", "k_layer")
            self.alpha = conf.getfloat("HPN", "alpha")
            self.edge_drop = conf.getfloat("HPN", "edge_drop")
            self.patience = conf.getint('HPN', 'patience')
            self.max_epoch = conf.getint('HPN', 'max_epoch')
            self.mini_batch_flag = conf.getboolean("HPN", "mini_batch_flag")

        elif self.model_name == 'SimpleHGN':
            self.weight_decay = conf.getfloat("SimpleHGN", "weight_decay")
            self.lr = conf.getfloat("SimpleHGN", "lr")
            self.max_epoch = conf.getint("SimpleHGN", "max_epoch")
            self.seed = conf.getint("SimpleHGN", "seed")
            self.patience = conf.getint("SimpleHGN", "patience")
            self.edge_dim = conf.getint("SimpleHGN", "edge_dim")
            self.slope = conf.getfloat("SimpleHGN", "slope")
            self.feats_drop_rate = conf.getfloat("SimpleHGN", "feats_drop_rate")
            self.num_heads = conf.getint("SimpleHGN", "num_heads")
            self.hidden_dim = conf.getint("SimpleHGN", "hidden_dim")
            self.num_layers = conf.getint("SimpleHGN", "num_layers")
            self.beta = conf.getfloat("SimpleHGN", "beta")
            self.residual = conf.getboolean("SimpleHGN", "residual")
            self.mini_batch_flag = conf.getboolean("SimpleHGN", "mini_batch_flag")
            self.fanout = conf.getint("SimpleHGN", "fanout")
            self.batch_size = conf.getint("SimpleHGN", "batch_size")
            self.use_uva = conf.getboolean("SimpleHGN", "use_uva")

        elif self.model_name == 'HetSANN':
            self.lr = conf.getfloat("HetSANN", "lr")
            self.weight_decay = conf.getfloat("HetSANN", "weight_decay")
            self.dropout = conf.getfloat("HetSANN", "dropout")
            self.seed = conf.getint("HetSANN", "seed")
            self.hidden_dim = conf.getint("HetSANN", "hidden_dim")
            self.num_layers = conf.getint("HetSANN", "num_layers")
            self.num_heads = conf.getint("HetSANN", "num_heads")
            self.max_epoch = conf.getint("HetSANN", "max_epoch")
            self.patience = conf.getint("HetSANN", "patience")
            self.slope = conf.getfloat("HetSANN", "slope")
            self.residual = conf.getboolean("HetSANN", "residual")
            self.mini_batch_flag = conf.getboolean("HetSANN", "mini_batch_flag")
            self.batch_size = conf.getint("HetSANN", "batch_size")
            self.fanout = conf.getint("HetSANN", "fanout")
            self.use_uva = conf.getboolean("HetSANN", "use_uva")

        elif self.model_name == 'ieHGCN':
            self.weight_decay = conf.getfloat("ieHGCN", "weight_decay")
            self.lr = conf.getfloat("ieHGCN", "lr")
            self.max_epoch = conf.getint("ieHGCN", "max_epoch")
            self.seed = conf.getint("ieHGCN", "seed")
            self.attn_dim = conf.getint("ieHGCN", "attn_dim")
            self.num_layers = conf.getint("ieHGCN", "num_layers")
            self.mini_batch_flag = conf.getboolean("ieHGCN", "mini_batch_flag")
            self.fanout = conf.getint("ieHGCN", "fanout")
            self.batch_size = conf.getint("ieHGCN", "batch_size")
            self.hidden_dim = conf.getint("ieHGCN", "hidden_dim")
            self.out_dim = conf.getint("ieHGCN", "out_dim")
            self.patience = conf.getint("ieHGCN", "patience")
            self.bias = conf.getboolean("ieHGCN", "bias")
            self.batchnorm = conf.getboolean("ieHGCN", "batchnorm")
            self.dropout = conf.getfloat("ieHGCN", "dropout")

        elif self.model_name == 'RGAT':
            self.weight_decay = conf.getfloat("RGAT", "weight_decay")
            self.lr = conf.getfloat("RGAT", "lr")
            self.max_epoch = conf.getint("RGAT", "max_epoch")
            self.seed = conf.getint("RGAT", "seed")
            self.num_layers = conf.getint("RGAT", "num_layers")
            self.mini_batch_flag = False
            self.hidden_dim = conf.getint("RGAT", "hidden_dim")
            self.in_dim = conf.getint("RGAT", "in_dim")
            self.patience = conf.getint("RGAT", "patience")
            self.num_heads = conf.getint("RGAT", "num_heads")
            self.dropout = conf.getfloat("RGAT", "dropout")
            self.out_dim = conf.getint("RGAT", "out_dim")

        if hasattr(self, 'device'):
            self.device = th.device(self.device)
        elif gpu == -1:
            self.device = th.device('cpu')
        elif gpu >= 0:
            if not th.cuda.is_available():
                self.device = th.device('cpu')
                warnings.warn("cuda is unavailable, the program will use cpu instead. please set 'gpu' to -1.")
            else:
                self.device = th.device('cuda', int(gpu))

        if getattr(self, 'use_uva', None):  # use_uva is set True
            self.use_uva = False
            warnings.warn("'use_uva' is only available when using cuda. please set 'use_uva' to False.")

    def __repr__(self):
        return '[Config Info]\tModel: {},\tTask: {},\tDataset: {}'.format(self.model_name, self.task, self.dataset)
