# -*- coding: utf-8 -*-
# @Time    : 17/04/2023 18:48
# @Function:
import os
import pickle
import sys
import logging

from util.local_models import local

DEV_MODE = True  # simplify the src.federatedscope re-setup everytime we change
# the source codes of src.federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

from src.federatedscope.core.cmd_args import parse_args, parse_client_cfg
from src.federatedscope.core.auxiliaries.data_builder import get_data
from src.federatedscope.core.auxiliaries.utils import setup_seed
from src.federatedscope.core.auxiliaries.logging import update_logger
from src.federatedscope.core.auxiliaries.worker_builder import get_client_cls, \
    get_server_cls
from src.federatedscope.core.configs.config import global_cfg, CfgNode
from src.federatedscope.core.auxiliaries.runner_builder import get_runner

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

root_logger = logging.getLogger("src.federatedscope")
if __name__ == '__main__':
    init_cfg = global_cfg.clone()
    args = parse_args()
    args.init_path = os.path.join(os.path.dirname(__file__))
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)
    # init_cfg.init_path = args.init_path
    init_cfg.data.root = os.path.abspath(init_cfg.data.root)
    # setting output_path

    res_path = f'results_batch_{init_cfg.data.anomaly_type}'
    init_cfg.outdir = (f"{res_path}/{init_cfg.federate.method}_{init_cfg.model.type}")
    print(init_cfg.outdir)

    init_cfg.expname = f"{init_cfg.federate.client_num}_{init_cfg.data.type}_{init_cfg.train.optimizer.lr}_" \
                       f"{init_cfg.dataloader.batch_size}_{init_cfg.fedsageplus.fedgen_epoch}_{init_cfg.train.local_update_steps}_" \
                       f"{init_cfg.federate.total_round_num}_{init_cfg.fedsageplus.a}_{init_cfg.fedsageplus.b}_{init_cfg.fedsageplus.c}"
    print(init_cfg.expname)
    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # load clients' cfg file
    if args.client_cfg_file:
        client_cfgs = CfgNode.load_cfg(open(args.client_cfg_file, 'r'))
        # client_cfgs.set_new_allowed(True)
        client_cfgs.merge_from_list(client_cfg_opt)
    else:
        client_cfgs = None
    # federated dataset might change the number of clients
    # thus, we allow the creation procedure of dataset to modify the global
    # cfg object

    # data_path = f'data/{init_cfg.data.type}/data_{init_cfg.federate.client_num}_{init_cfg.seed}.pkl'
    # modified_cfg_path = f'data/{init_cfg.data.type}/modified_cfg_{init_cfg.federate.client_num}_{init_cfg.seed}.pkl'
    # if os.path.exists(data_path) and os.path.exists(modified_cfg_path):
    #     with open(data_path, "rb") as f:
    #         data = pickle.load(f)
    #     with open(modified_cfg_path, "rb") as f:
    #         modified_cfg = pickle.load(f)
    # else:
    data, modified_cfg = get_data(config=init_cfg.clone(),
                                  client_cfgs=client_cfgs)
        #store
        # pickle.dump(data, open(data_path, 'wb'))
        # pickle.dump(modified_cfg, open(modified_cfg_path, 'wb'))

    init_cfg.merge_from_other_cfg(modified_cfg)

    init_cfg.freeze(inform=False)
    # logger = logging.getLogger(__name__)
    if init_cfg.federate.method == 'local':
        local(data,init_cfg)
    else:
        runner = get_runner(data=data,
                            server_class=get_server_cls(init_cfg),
                            client_class=get_client_cls(init_cfg),
                            config=init_cfg.clone(),
                            client_configs=client_cfgs)
        _ = runner.run()
    root_logger.info("Done!")