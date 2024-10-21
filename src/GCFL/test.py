# -*- coding: utf-8 -*-
# @Time    : 22/02/2023 16:09
# @Function:
# -*- coding: utf-8 -*-
# @Time    : 22/02/2023 15:46
# @Function:
import os
import argparse
import random
import torch
from pathlib import Path
import copy
import numpy as np
from src.setupGC import *
import yaml
from src.training import *
from parser import Parser,add_argu

def process_selftrain(clients, server, local_epoch):
    print("Self-training ...")
    df = pd.DataFrame()
    allAccs = run_selftrain_GC(clients, server, local_epoch)
    for k, v in allAccs.items():
        df.loc[k, [f'train_acc', f'val_acc', f'test_acc']] = v
    print(df)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_selftrain_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_selftrain_GC{suffix}.csv')
    df.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_fedavg(clients, server):
    print("\nDone setting up FedAvg devices.")

    print("Running FedAvg ...")
    frame = run_fedavg(clients, server, args.num_rounds, args.local_epoch, samp=None)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedavg_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedavg_GC{suffix}.csv')
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_fedprox(clients, server, mu):
    print("\nDone setting up FedProx devices.")

    print("Running FedProx ...")
    frame = run_fedprox(clients, server, args.num_rounds, args.local_epoch, mu, samp=None)
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_fedprox_mu{mu}_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_fedprox_mu{mu}_GC{suffix}.csv')
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_gcfl(clients, server):
    print("\nDone setting up GCFL devices.")
    print("Running GCFL ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_gcfl_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcfl_GC{suffix}.csv')

    frame = run_gcfl(clients, server, args.num_rounds, args.local_epoch, EPS_1, EPS_2)
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_gcflplus(clients, server):
    print("\nDone setting up GCFL devices.")
    print("Running GCFL plus ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_gcflplus_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcflplus_GC{suffix}.csv')

    frame = run_gcflplus(clients, server, args.num_rounds, args.local_epoch, EPS_1, EPS_2, args.seq_length, args.standardize)
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")


def process_gcflplusdWs(clients, server):
    print("\nDone setting up GCFL devices.")
    print("Running GCFL plus with dWs ...")

    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_gcflplusDWs_GC{suffix}.csv')
    else:
        outfile = os.path.join(outpath, "repeats", f'{args.repeat}_accuracy_gcflplusDWs_GC{suffix}.csv')

    frame = run_gcflplus_dWs(clients, server, args.num_rounds, args.local_epoch, EPS_1, EPS_2, args.seq_length, args.standardize)
    frame.to_csv(outfile)
    print(f"Wrote to file: {outfile}")

def Outpath(args):
    outbase = os.path.join(args.outbase, f'seqLen{args.seq_length}')
    if args.overlap and args.standardize:
        outpath = os.path.join(outbase, f"standardizedDTW/oneDS-overlap")
    elif args.overlap:
        outpath = os.path.join(outbase, f"oneDS-overlap")
    elif args.standardize:
        outpath = os.path.join(outbase, f"standardizedDTW/oneDS-nonOverlap")
    else:
        outpath = os.path.join(outbase, f"oneDS-nonOverlap")
    outpath = os.path.join(outpath, f'{args.data_group}-{args.num_clients}clients', f'eps_{EPS_1}_{EPS_2}')
    Path(outpath).mkdir(parents=True, exist_ok=True)
    print(f"Output Path: {outpath}")
    return outpath

if __name__ == '__main__':
    args = Parser(description='Explainer').args
    args.device = torch.device(args.device) # "cuda" if torch.cuda.is_available() else "cpu"
    # set seeds
    seed_dataSplit = 123
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # parameter setting
    config_dir = Path('./configs')
    args = add_argu(args,config_dir)
    EPS_1 = args.epsilon1
    EPS_2 = args.epsilon2
    outpath = Outpath(args)

    """ distributed one dataset to multiple clients """

    if not args.convert_x:
        """ using original features """
        suffix = ""
        print("Preparing data (original features) ...")
    else:
        """ using node degree features """
        suffix = "_degrs"
        print("Preparing data (one-hot degree features) ...")

    if args.repeat is not None:
        Path(os.path.join(outpath, 'repeats')).mkdir(parents=True, exist_ok=True)

    print("Load Dataset")
    if args.dataset_setting == 'oneDS':
        splitedData, df_stats = prepareData_oneDS(args.datapath, args.data_group, num_client=args.num_clients, batchSize=args.batch_size,
                                                      convert_x=args.convert_x, seed=seed_dataSplit, overlap=args.overlap)
    else:
        splitedData, df_stats = prepareData_multiDS(args.datapath, args.data_group, args.batch_size,
                                                        convert_x=args.convert_x, seed=seed_dataSplit)

    print("Done")

    # save statistics of data on clients
    if args.repeat is None:
        outf = os.path.join(outpath, f'stats_trainData{suffix}.csv')
    else:
        outf = os.path.join(outpath, "repeats", f'{args.repeat}_stats_trainData{suffix}.csv')
    df_stats.to_csv(outf)
    print(f"Wrote to {outf}")

    init_clients, init_server, init_idx_clients = setup_devices(splitedData, args)
    print("\nDone setting up devices.")

    process_selftrain(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), local_epoch=50)
    process_fedavg(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))
    process_fedprox(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server), mu=0.01)
    process_gcfl(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))
    process_gcflplus(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))
    process_gcflplusdWs(clients=copy.deepcopy(init_clients), server=copy.deepcopy(init_server))

