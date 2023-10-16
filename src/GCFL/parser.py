# -*- coding: utf-8 -*-
# @Time    : 22/02/2023 15:48
# @Function:
# -*- coding: utf-8 -*-
import argparse
import yaml
class Parser():
    def __init__(self, description):
        '''
           arguments parser
        '''
        self.parser = argparse.ArgumentParser(description=description)
        self.args = None
        self._parse()

    def _parse(self):
        ### ------------------  device     -------------- ###
        self.parser.add_argument('--device', type=str, default='cuda:2', help='The device to run the program')

        ### ------------------  data       ---------------###
        self.parser.add_argument('--repeat', help='index of repeating;',
                        type=int, default=1)
        self.parser.add_argument('--data_group', help='specify the group of datasets',
                        type=str, default='IMDB-BINARY')
        ### ------------------  FL         ---------------###
        self.parser.add_argument('--dataset_setting',type=str, default='oneDS')

        ### -------------------- data_common_config -------####
        self.parser.add_argument('--datapath', type=str, default='./data',
                            help='The input path of data.')
        self.parser.add_argument('--outbase', type=str, default='./outputs',
                            help='The base path for outputting.')
        self.parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features',
                            type=bool, default=False)
        self.parser.add_argument('--overlap', help='whether clients have overlapped data',
                            type=bool, default=False)
        self.parser.add_argument('--standardize', help='whether to standardize the distance matrix',
                            type=bool, default=False)
        ### -------------------- model_common_config -------####
        self.parser.add_argument('--num_rounds', type=int, default=200,
                            help='number of rounds to simulate;')
        self.parser.add_argument('--local_epoch', type=int, default=1,
                            help='number of local epochs;')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
        self.parser.add_argument('--nlayer', type=int, default=3,
                            help='Number of GINconv layers')
        self.parser.add_argument('--hidden', type=int, default=64,
                            help='Number of hidden units.')
        self.parser.add_argument('--dropout', type=float, default=0.5,
                            help='Dropout rate (1 - keep probability).')
        self.parser.add_argument('--batch_size', type=int, default=128,
                            help='Batch size for node classification.')
        self.parser.add_argument('--seed', help='seed for randomness;',
                            type=int, default=10)
        ###-------------------- important para --------------#######
        self.parser.add_argument('--num_clients', help='number of clients',type=int)
        self.parser.add_argument('--seq_length', help='the length of the gradient norm sequence',type=int)
        self.parser.add_argument('--epsilon1', help='the threshold epsilon1 for GCFL',type=float)
        self.parser.add_argument('--epsilon2', help='the threshold epsilon2 for GCFL',type=float)

        self.parser.add_argument('--num_repeat', type=int, help='number of repeating rounds to simulate;')
        self.parser.add_argument('--lr', type=float,help='learning rate for inner solver;')

        self.args = self.parser.parse_args()

def add_argu(args,config_dir):
    # read parameters config
    print(f'[INFO] Running {args.dataset_setting} on {args.data_group} ')
    local_config_name = f'{args.dataset_setting}.yml'
    local_config = yaml.safe_load((config_dir / local_config_name).open('r'))
    dataset_config = local_config[f'{args.data_group}_config']
    model_common_config = local_config['model_common_config']
    args.num_clients = dataset_config['num_clients']
    args.epsilon1 = dataset_config['epsilon1']
    args.epsilon2 = dataset_config['epsilon2']
    args.seq_length = dataset_config['seq_length']
    args.num_repeat = model_common_config['num_repeat']
    args.lr = model_common_config['lr']
    return args