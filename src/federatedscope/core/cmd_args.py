import argparse
import sys
from src.federatedscope.core.configs.config import global_cfg
import sys
import os
import argparse

current_file_name = __file__
current_dir=os.path.dirname(os.path.dirname(os.path.abspath(current_file_name)))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from src.dgld.utils.common import tab_printer, loadargs_from_json
from src.dgld.utils.common_params import IN_FEATURE_MAP,NUM_NODES_MAP
# DOMINANT
from src.dgld.models.DOMINANT import set_subargs as dominant_set_args
from src.dgld.models.DOMINANT import get_subargs as dominant_get_args
#AnomalyDAE
from src.dgld.models.AnomalyDAE import set_subargs as anomalydae_set_args
from src.dgld.models.AnomalyDAE import get_subargs as anomalydae_get_args
# ComGA
from src.dgld.models.ComGA import set_subargs as comga_set_args
from src.dgld.models.ComGA import get_subargs as comga_get_args
# DONE
from src.dgld.models.DONE import set_subargs as done_set_args
from src.dgld.models.DONE import get_subargs as done_get_args
# AdONE
from src.dgld.models.AdONE import set_subargs as adone_set_args
from src.dgld.models.AdONE import get_subargs as adone_get_args
# CONAD
from src.dgld.models.CONAD import set_subargs as conad_set_args
from src.dgld.models.CONAD import get_subargs as conad_get_args
# ALARM
from src.dgld.models.ALARM import set_subargs as alarm_set_args
from src.dgld.models.ALARM import get_subargs as alarm_get_args
# ONE
from src.dgld.models.ONE import set_subargs as one_set_args
from src.dgld.models.ONE import get_subargs as one_get_args
# GAAN
from src.dgld.models.GAAN import set_subargs as gaan_set_args
from src.dgld.models.GAAN import get_subargs as gaan_get_args
# GUIDE
from src.dgld.models.GUIDE import set_subargs as guide_set_args
from src.dgld.models.GUIDE import get_subargs as guide_get_args
# CoLA
from src.dgld.models.CoLA import set_subargs as cola_set_args
from src.dgld.models.CoLA import get_subargs as cola_get_args
# SL-GAD
from src.dgld.models.SLGAD import set_subargs as slgad_set_args
from src.dgld.models.SLGAD import get_subargs as slgad_get_args
# AAGNN
from src.dgld.models.AAGNN import set_subargs as aagnn_set_args
from src.dgld.models.AAGNN import get_subargs as aagnn_get_args
# ANEMONE
from src.dgld.models.ANEMONE import set_subargs as anemone_set_args
from src.dgld.models.ANEMONE import get_subargs as anemone_get_args
# GCNAE
from src.dgld.models.GCNAE import set_subargs as gcnae_set_args
from src.dgld.models.GCNAE import get_subargs as gcnae_get_args
#MLPAE
from src.dgld.models.MLPAE import set_subargs as mlpae_set_args
from src.dgld.models.MLPAE import get_subargs as mlpae_get_args
#SCAN
from src.dgld.models.SCAN import set_subargs as scan_set_args
from src.dgld.models.SCAN import get_subargs as scan_get_args

# set args
models_set_args_map = {
    "DOMINANT": dominant_set_args,
    "AnomalyDAE": anomalydae_set_args,
    "ComGA": comga_set_args,
    "DONE": done_set_args,
    "AdONE": adone_set_args,
    "CONAD": conad_set_args,
    "ALARM": alarm_set_args,
    "ONE": one_set_args,
    "GAAN": gaan_set_args,
    "GUIDE": guide_set_args,
    "CoLA": cola_set_args,
    "SLGAD": slgad_set_args,
    "AAGNN": aagnn_set_args,
    "ANEMONE": anemone_set_args,
    "GCNAE": gcnae_set_args,
    "MLPAE": mlpae_set_args,
    "SCAN": scan_set_args
}
# get args
models_get_args_map = {
    "DOMINANT": dominant_get_args,
    "AnomalyDAE": anomalydae_get_args,
    "ComGA": comga_get_args,
    "DONE": done_get_args,
    "AdONE": adone_get_args,
    "CONAD": conad_get_args,
    "ALARM": alarm_get_args,
    "ONE": one_get_args,
    "GAAN": gaan_get_args,
    "GUIDE":guide_get_args,
    "CoLA": cola_get_args,
    "SLGAD":slgad_get_args,
    "AAGNN": aagnn_get_args,
    "ANEMONE": anemone_get_args,
    "GCNAE": gcnae_get_args,
    "MLPAE": mlpae_get_args,
    "SCAN": scan_get_args
}

# def parse_args(args=None):
#     parser = argparse.ArgumentParser(description='src.federatedscope',
#                                      add_help=False)
#     parser.add_argument('--cfg',
#                         dest='cfg_file',
#                         # default='gfl/baseline/fedavg_gnn_node_fullbatch_citation.yaml',
#                         help='Config file path',
#                         required=False,
#                         type=str)
#     parser.add_argument('--client_cfg',
#                         dest='client_cfg_file',
#                         help='Config file path for clients',
#                         required=False,
#                         default=None,
#                         type=str)
#     parser.add_argument('--dataset',
#                         type=str,
#                         default='Cora',
#                         help='Dataset used in the experiment')
#     parser.add_argument('--feat_dim',
#                         type=int,
#                         default=1433,
#                         help='number of features dimension. Defaults to 1000.')
#     parser.add_argument('--num_nodes',
#                         type=int,
#                         default=2708,
#                         help='number of nodes. Defaults to 2708.')
#     parser.add_argument('--data_path',
#                         type=str,
#                         default='src/src.dgld/data/',
#                         help='data path')
#     parser.add_argument('--device',
#                         type=str,
#                         default='0',
#                         help='ID(s) of gpu used by cuda')
#     parser.add_argument('--seed',
#                         type=int,
#                         default=4096,
#                         help='Random seed. Defaults to 4096.')
#     parser.add_argument('--save_path',
#                         type=str,
#                         help='save path of the result')
#     parser.add_argument('--exp_name',
#                         type=str,
#                         help='exp_name experiment identification')
#     parser.add_argument('--runs',
#                         type=int,
#                         default=1,
#                         help='The number of runs of task with same parmeter,If the number of runs is not 1, \
#                                 we will randomly generate different seeds to calculate the variance')
#     parser.add_argument(
#         '--help',
#         nargs="?",
#         const="all",
#         default="",
#     )
#     parser.add_argument('opts',
#                         help='See src.federatedscope/core/configs for all options',
#                         default=None,
#                         nargs=argparse.REMAINDER)
#     # get dataset
#     arg_list = sys.argv[1:]
#     if '--dataset' in arg_list:
#         idx = arg_list.index('--dataset') + 1
#         dataset = arg_list[idx]
#     elif any(map(lambda x: x.startswith('--dataset='), arg_list)):
#         dataset = [x.split("=")[-1] for x in arg_list if x.startswith('--dataset=')]
#         dataset = dataset[0]
#     else:
#         dataset = parser.get_default('dataset')
#
#     # set default feat_dim and num_nodes
#     if dataset in IN_FEATURE_MAP.keys():
#         parser.set_defaults(feat_dim=IN_FEATURE_MAP[dataset], num_nodes=NUM_NODES_MAP[dataset])
#
#     subparsers = parser.add_subparsers(dest="model", help='sub-command help')
#
#     # set sub args
#     for _model, set_arg_func in models_set_args_map.items():
#         sub_parser = subparsers.add_parser(
#             _model, help=f"Run anomaly detection on {_model}")
#         set_arg_func(sub_parser)
#
#         # set best args
#         fp = f'src/src.dgld/config/{_model}.json'
#         if os.path.exists(fp):
#             best_config = loadargs_from_json(fp)
#             sub_parser.set_defaults(**best_config.get(dataset, {}))
#
#     args_dict, args = models_get_args_map[args.model](args)
#     parse_res = parser.parse_args(args)
#     init_cfg = global_cfg.clone()
#     # when users type only "Fed_example.py" or "Fed_example.py help"
#     # print(parse_res.cfg)
#     if (len(sys.argv) == 1 and parse_res.cfg==None) or parse_res.help == "all":
#         parser.print_help()
#         init_cfg.print_help()
#         sys.exit(1)
#     elif hasattr(parse_res, "help") and isinstance(
#             parse_res.help, str) and parse_res.help != "":
#         init_cfg.print_help(parse_res.help)
#         sys.exit(1)
#     elif hasattr(parse_res, "help") and isinstance(
#             parse_res.help, list) and len(parse_res.help) != 0:
#         for query in parse_res.help:
#             init_cfg.print_help(query)
#         sys.exit(1)
#
#     return parse_res,args_dict
#

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='federatedscope',
                                     add_help=False)
    parser.add_argument('--cfg',
                        dest='cfg_file',
                        # default='gfl/baseline/fedavg_gnn_node_fullbatch_citation.yaml',
                        help='Config file path',
                        required=False,
                        type=str)
    parser.add_argument('--client_cfg',
                        dest='client_cfg_file',
                        help='Config file path for clients',
                        required=False,
                        default=None,
                        type=str)
    parser.add_argument(
        '--help',
        nargs="?",
        const="all",
        default="",
    )
    parser.add_argument('opts',
                        help='See federatedscope/core/configs for all options',
                        default=None,
                        nargs=argparse.REMAINDER)
    parse_res = parser.parse_args(args)
    init_cfg = global_cfg.clone()
    # when users type only "main.py" or "main.py help"
    # print(parse_res.cfg)
    if (len(sys.argv) == 1 and parse_res.cfg==None) or parse_res.help == "all":
        parser.print_help()
        init_cfg.print_help()
        sys.exit(1)
    elif hasattr(parse_res, "help") and isinstance(
            parse_res.help, str) and parse_res.help != "":
        init_cfg.print_help(parse_res.help)
        sys.exit(1)
    elif hasattr(parse_res, "help") and isinstance(
            parse_res.help, list) and len(parse_res.help) != 0:
        for query in parse_res.help:
            init_cfg.print_help(query)
        sys.exit(1)

    return parse_res




def parse_client_cfg(arg_opts):
    """
    Arguments:
        arg_opts: list pairs of arg.opts
    """
    client_cfg_opts = []
    i = 0
    while i < len(arg_opts):
        if arg_opts[i].startswith('client'):
            client_cfg_opts.append(arg_opts.pop(i))
            client_cfg_opts.append(arg_opts.pop(i))
        else:
            i += 1
    return arg_opts, client_cfg_opts
