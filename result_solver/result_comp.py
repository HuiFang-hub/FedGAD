# -*- coding: utf-8 -*-
# @Time    : 18/05/2023 14:15
# @Function:
# -*- coding: utf-8 -*-
# @Time    : 17/04/2023 16:31
# @Function:
import matplotlib.pyplot as plt
import re
import os
import glob
import pandas as pd
import ast
import numpy as np
import logging

from util.vision import plot_auc


def role_result_match(role,text,metric):
    if role == "Client":
        # Client
        matches = re.findall(
            r"{'Role': 'Client #([1-3])', 'Round': \d+, 'Results_raw': {'train_avg_loss': (\d+\.\d+), "
            r"'train_roc_auc': (\d+\.\d+),'train_acc': (\d+\.\d+),'train_recall': (\d+\.\d+) 'train_total': (\d+)}}",
            text)

        result = {}
        for match in matches:

            role = "Client" + match[0]

                # result[role] = {"train_avg_loss": [], "train_roc_auc": []}
            for i,m in enumerate(metric):
                if m not in result:
                    result[m] = {role: []}
                if role not in result[m]:
                    result[m][role]=[]
                result[m][role].append(float(match[i+1]))
    else:
        matches=  re.findall(
            r"{'Role': 'Server #', 'Round': \d+, 'Results_raw': {'test_roc_auc': (\d+\.\d+), 'test_total': (\d+)}}",
            text)
        result = {}
        for match in matches:
            for i,m in enumerate(metric):
                if m not in result:
                    result[m]= []
                result[m].append(float(match[i]))

    return result

def get_path(dir_path):
    conditions = []
    if "exp" in dir_path:
        para_name = ['federate.client_num', 'data.type', 'train.optimizer.lr', 'dataloader.batch_size',
                     'fedsageplus.fedgen_epoch', 'train.local_update_steps', 'federate.total_round_num',
                     'fedsageplus.a',
                     'fedsageplus.b', 'fedsageplus.c']
    else:
        para_name = ['federate.client_num', 'data.type', 'train.optimizer.lr', 'dataloader.batch_size',
                     'fedsageplus.fedgen_epoch', 'train.local_update_steps', 'federate.total_round_num', 'data.splitter',
                     'data.splitter_args']

    dirs = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
    for dir in dirs:
        #dirs: all subfolder


        # extract name of parameters from subfolder name
        # para_name = ['federate.client_num','data.type','train.optimizer.lr','dataloader.batch_size',
        #              'fedsageplus.fedgen_epoch','train.local_update_steps','federate.total_round_num','fedsageplus.a','fedsageplus.b','fedsageplus.c']
        condition = {}
        subdir_path = os.path.join(dir_path, dir)
        # method_name = dir.split('_')
        #
        condition["method_name"] = dir
        condition["method_path"] = subdir_path # './exp/fedavg_cola'
        #
        sub_dirs = [name for name in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, name))]
        #
        #
        # for root1, dirs1, files1 in os.walk(subdir_path):
        paras_paths = []
        paras_dict= []
        for p_v in sub_dirs:
            paras_path = os.path.join(subdir_path, p_v)
            paras_paths.append(paras_path)
            paras_val = p_v.split('_')
            paras = {}
            for n,para_val in zip(para_name,paras_val):
                paras[n]=para_val
            paras_dict.append(paras)
        condition["paras_paths"] = paras_paths
        condition["paras_dict"] = paras_dict
        conditions.append(condition)
    print(conditions)
    return conditions

def local_extrat(text,m):
    # process client data
    result_dict = {}
    finish = 0

    # if finish
    if "Done!" in text:
        finish = 1
        # graph structure
        para = ['num_nodes','sum_nodes','average_num_nodes','num_edges','sum_edges','average_num_edges']
        pattern = r'num_nodes:\[(.*?)\] sum_nodes:(\d+) average_num_nodes:(\d+\.\d+)[\s\S]*?num_edges:\[(.*?)\] sum_edges:(\d+) average_num_edges:(\d+\.\d+)'
        matches = re.findall(pattern, text)
        for match in matches:
            for p,v in zip(para,match):
                result_dict[p] = [(float(i)) for i in v.split(', ')]
        # loss
        client_ids = []
        pattern = r'Client:\s(\d+).*Loss:\s([.\d]+).*AUC:\s([.\d]+)'
        for match in re.findall(pattern, text):
            client_id = match[0]
            loss = match[1]
            auc = match[2]

            # 将数据存储到字典中
            if client_id not in result_dict:
                result_dict[client_id] = {'loss': [], 'auc': []}
                client_ids.append(client_id)
            result_dict[client_id]['loss'].append(loss)
            result_dict[client_id]['auc'].append(auc)
        result_dict["client_num"] = client_ids
        #results  ".join([f"'train_{key}': array\((\[.*?\])\)" for key in train_m[-2:]]
        pattern = r"(\d+): \{" + ", ".join([f"'{key}': (.*?)" for key in m[:-2]]) +", "+ r", ".join([f"'{key}': array\((\[.*?\])\)" for key in m[-2:]])
        # print(pattern)
        #r", 'fpr': array\(\[(.*?)\]\), 'tpr': array\(\[(.*?)\]\).*?\}"
        results_str = re.findall(pattern, text, re.DOTALL)

        result_avg = {key: 0 for key in m}
        values = {}
        for index,r in enumerate(results_str):
            key = r[0]
            for i in range(len(m)):
                if ',' in r[i + 1]:
                    values[m[i]] = ast.literal_eval(r[i + 1]) #[float(x.strip()) for x in r[i + 1].split(',')]
                else:
                    values[m[i]] = float(r[i+1].strip())
                    if r[0] != '0':
                        # result_avg[m[i]] += values[m[i]] * int(result_dict['num_nodes'][index]) / int(
                        #     result_dict['sum_nodes'][0])
                        result_avg[m[i]] += values[m[i]] / (len(result_dict["client_num"] )-1)
            result_dict[key].update(values)
        result_dict['avg'] = result_avg
        result_dict["client_num"].append('avg')
    return finish,result_dict

def loc_comp_avg(results,metric):
    client_ids = results[0]['client_num']
    result_dict = {}
    for client_id in client_ids:
        res_id = {key: [] for key in metric}
        cnt = 1
        for result in results:
            res = result[client_id]
            for m in metric:
                # test = res_id[m]
                res_id[m].append(res[m])
            cnt += 1
        m_dict = {}
        for k,v in res_id.items():
            mean = np.mean(v)
            stddev = np.std(v)
            # m_dict[k+"_mean"] = mean
            # m_dict[k + "_stddev"] = stddev
            m_dict[k.replace('test_', '')] = str(round(mean*100,2)) + '$\pm$' + str(round(stddev*10,2))
            # if client_id!= 0:
            #     result_avg[k] += ( np.sum(v)) * int(results[0]['num_nodes'][index])/int(results[0]['sum_nodes'][0])
        result_dict[client_id] = m_dict
    return result_dict


def comp_auc(results,condition,line_dict):
    # client_ids = results.keys()
    metric = ['fpr','tpr']
    # result_dict = {}

    if 'local' in condition:
        client_ids = results[0]['client_num']
        for client_id in client_ids[:-1]:
            res_id = {key: [] for key in metric}
            cnt = 1
            for result in results:
                res = result[client_id]
                for m in metric:
                    # test = res_id[m]
                    res_id[m].append(res[m])
                cnt += 1
            for f,t in zip(res_id['fpr'],res_id['tpr']):
                if client_id == '0':
                    line_dict['method_name'] += ['Global' for _ in range(len(f))]
                else:
                    line_dict['method_name'] += [f'loc_{client_id}' for _ in range(len(f))]
                    # line_dict['method_name'] += [f'{condition}' for _ in range(len(f))]
                line_dict['fpr'] += f
                line_dict['tpr'] += t
    else:
        res_id = {key: [] for key in metric}
        for result in results:
            res = result[0]
            for m in metric:
                # test = res_id[m]
                res_id[m].append(res[m])
        for f, t in zip(res_id['fpr'], res_id['tpr']):
            # line_dict['method_name'] += [f'loc_{client_id}' for _ in range(len(f))]
            if 'fedavg' in condition:
                line_dict['method_name'] += ['FedGAD' for _ in range(len(f))]
            else:
                line_dict['method_name'] += ['FedGAD+' for _ in range(len(f))]
            line_dict['fpr'] += f
            line_dict['tpr'] += t


    # line_df = pd.DataFrame(line_dict)
    # plot_lineplot(line_df)
    # print("test")
    return line_dict





        # for k, v in res_id.items():
        #     mean = np.mean(v)
        #     stddev = np.std(v)
        #     # m_dict[k+"_mean"] = mean
        #     # m_dict[k + "_stddev"] = stddev
        #     m_dict[k.replace('test_', '')] = str(round(mean * 100, 2)) + '$\pm$' + str(round(stddev * 10, 2))
        #     # if client_id!= 0:
        #     #     result_avg[k] += ( np.sum(v)) * int(results[0]['num_nodes'][index])/int(results[0]['sum_nodes'][0])
        # result_dict[client_id] = m_dict
    # return result_dict

def fed_extrat(text,train_m,test_m,client_num):
    # process client data
    result_dict = {key: {} for key in range(client_num+1)}
    finish = 0

    # if finish
    if "Final" in text:
        finish = 1
        text = text.replace('\n', '')
        # client
        client_pattern = r"'Role': 'Client #(\d+)', 'Round': (\d+)," + \
                  r" 'Results_raw': {" + ", ".join([f"'train_{key}': (.*?)" for key in train_m[:-2]]) + \
                  r", ".join([f"'train_{key}': array\((\[.*?\])\)" for key in train_m[-2:]])
                  # r", 'train_fpr': array\((\[.*?\])\),\s*'train_tpr': array\((\[.*?\])\)"
        matchs = re.findall(client_pattern, text)
        for match in  matchs:
            key = int(match[0])
            values = {}
            values['round'] = match[1]
            for keym,value in zip(train_m,match[2:]):
                if keym =='train_total':
                    continue
                else:
                    try:
                        values[keym] = ast.literal_eval(value)
                        # print(result)
                    except (SyntaxError, ValueError):
                        continue
            result_dict[key] = values
        #Server
        server_pattern = r"'Role': 'Server #', 'Round': 'Final'," + \
                  r" 'Results_raw': {'server_global_eval': {" + ", ".join([f"'test_{key}': (.*?)" for key in test_m[:-2]]) + \
                  r", ".join([f"'test_{key}': array\((\[.*?\])\)" for key in test_m[-2:]])
        matchs = re.findall(server_pattern, text)[0]
        values = {}
        for keym,value in zip(test_m,matchs):
            values[keym] = ast.literal_eval(value)
        result_dict[0] = values


    return finish,result_dict

def fed_comp_avg(results,metric,client_num):

    client_ids = range(client_num+1)
    result_dict = {}
    # result_dict[0] = results[0]
    for client_id in client_ids:
        res_id = {key: [] for key in metric[:-2]}
        cnt = 1
        for result in results:
            res = result[client_id]
            for m in metric[:-2]:
                # test = res_id[m]
                res_id[m].append(res[m])
            cnt += 1
        m_dict = {}
        for k,v in res_id.items():
            mean = np.mean(v)
            stddev = np.std(v)
            # m_dict[k+"_mean"] = mean
            # m_dict[k + "_stddev"] = stddev
            m_dict[k] = str(round(mean*100,2)) + '$\pm$' + str(round(stddev*10,2))
            # if client_id!= 0:
            #     result_avg[k] += ( np.sum(v)) * int(results[0]['num_nodes'][index])/int(results[0]['sum_nodes'][0])
        result_dict[client_id] = m_dict
    return result_dict

def excel_res(conditions):
    my_df = pd.DataFrame()
    Metric = ['roc_auc', 'acc', 'recall', 'f1']
    param = 'batch_size'
    param_v = ["20", "50", "100"]  # ["50","100","200"]
    client_num = 10
    method = ['local_anemone', 'fedavg_anemone', 'fedsagegod_anemone']  # 'local_anemone','fedavg_anemone',
    # datset_name = ['amazon']
    for condition in conditions:
        my_dict = {}
        env_cnt = 0
        if condition['method_name'] in method:
            for para_path, paras_dict in zip(condition['paras_paths'], condition['paras_dict']):
                # client_num = int(paras_dict['federate.client_num'])
                data_name = paras_dict['data.type']
                # if data_name in datset_name:
                # if int(paras_dict['federate.client_num']) == client_num:
                params = paras_dict.keys()
                p = [s for s in params if param in s][0]
                for p_v in param_v:
                    if int(paras_dict['federate.client_num']) == client_num and paras_dict[p] == p_v:
                        print(para_path)
                        # logging.info(para_path)
                        # 3 round files
                        results = []
                        round_dirs = [name for name in os.listdir(para_path) if
                                      os.path.isdir(os.path.join(para_path, name))]
                        for round_name in round_dirs:
                            round_dir = os.path.join(para_path, round_name)
                            if 'local' in condition['method_name']:
                                if paras_dict['dataloader.batch_size'] == p_v:
                                    file_name = glob.glob(os.path.join(round_dir, '*.log'))
                                    with open(file_name[0], 'r') as f:
                                        log_content = f.read()
                                        # remove nan
                                        log_content = log_content.replace('nan', '0.5')
                                        m = ['test_roc_auc', 'test_acc', 'f1', 'test_recall', 'test_recall_macro',
                                             'test_recall_weight', 'prec_5',
                                             'prec_10', 'prec_20','fpr', 'tpr']
                                        # m = ['test_acc']
                                        finish, result = local_extrat(log_content, m)
                                else:
                                    finish = False
                            else:

                                file_name = glob.glob(os.path.join(round_dir, 'eval_results_raw.log'))
                                if file_name:
                                    with open(file_name[0], 'r') as f:
                                        log_content = f.read()
                                        log_content = log_content.replace('nan', '0.5')
                                        train_m = ['avg_loss', 'roc_auc', 'acc', 'f1', 'recall', 'recall_macro',
                                                   'recall_weight', 'prec_5', 'prec_10', 'prec_20', 'total', 'fpr',
                                                   'tpr']
                                        test_m = ['roc_auc', 'acc', 'f1', 'recall', 'recall_macro', 'recall_weight',
                                                  'prec_5', 'prec_10', 'prec_20', 'fpr', 'tpr']
                                        finish, result = fed_extrat(log_content, train_m, test_m, client_num)
                                else:
                                    finish = False
                            if finish:
                                results.append(result)
                            else:
                                continue
                        # calculate average
                        if results:
                            if 'local' in condition['method_name']:
                                result_dic = loc_comp_avg(results, m)
                                # result_dic = {k: v for k, v in result_dic.items() if k in ['0','avg']}
                            else:
                                result_dic = fed_comp_avg(results, test_m, client_num)
                                result_dic = result_dic[0]
                            result_df = pd.DataFrame.from_dict(result_dic, orient='index')
                            # logging.info(result_df.style.to_latex())
                            # print(result_df.style.to_latex())
                            # env_cnt+=1
                            method_name = condition['method_name'].split('_')[0]
                            if data_name == 'facebookpagepage-inj-before':
                                data_name = 'FacebookPP'
                            else:
                                lst = data_name.split('-')
                                if len(lst) > 3:
                                    data_name = lst[0].capitalize() + '-' + lst[1].upper()
                                else:
                                    data_name = lst[0].capitalize()
                            for m in Metric:
                                if 'roc_auc' in m:
                                    mm = 'AUC'
                                else:
                                    mm = m.capitalize()
                                if 'local' in method_name:
                                    my_df.loc['GlobalGAD' + '_' + p_v, data_name + '_' + mm] = result_df.loc["0", m]
                                    my_df.loc['LocalGAD' + '_' + p_v, data_name + '_' + mm] = result_df.loc['avg', m]
                                elif 'fedavg' in method_name:
                                    my_df.loc['FedGAD' + '_' + p_v, data_name + '_' + mm] = result_dic[m]
                                else:
                                    my_df.loc['FedGAD+' + '_' + p_v, data_name + '_' + mm] = result_dic[m]
                        print(my_df)

    row_order = ['LocalGAD', 'FedGAD', 'FedGAD+', 'GlobalGAD']
    index = sorted(my_df.index, key=lambda x: (row_order.index(x.split("_")[0]), int(x.split("_")[1])))
    my_df_sorted = my_df.reindex(index=index)
    col_order1 = ["Cora", "Citeseer",
                  "Twitch-PT", "Twitch-DE",
                  "FacebookPP", "Tfinance", "Amazon", "Yelp"]
    col_order2 = ["AUC", "Acc", "Recall", 'F1']
    col = sorted(list(my_df_sorted),
                 key=lambda x: (col_order1.index(x.split("_")[0]), (col_order2.index(x.split("_")[1]))))

    my_df_sorted = my_df_sorted.reindex(columns=col)
    print(my_df_sorted)
    my_df_sorted.to_excel(dir_path + f'/results_{client_num}_{param}.xlsx')
    logging.info(my_df_sorted.style.to_latex())

if __name__ == '__main__':
    dir_path = './results_batchSize_cs'
    # conditions = ["local_cola","fedavg_cola","fedsagegod_cola"]
    logging.basicConfig(filename=dir_path + '/result.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    conditions = get_path(dir_path)
    # excel
    # excel_res(conditions)
    # auc curve


    my_df = pd.DataFrame()
    Metric = ['roc_auc', 'acc', 'recall', 'f1']
    param = 'batch_size'
    param_v = ["100"]  # ["50","100","200"]
    client_num = 3
    method = ['local_anemone', 'fedavg_anemone', 'fedsagegod_anemone']  # 'local_anemone','fedavg_anemone',
    dataset_name = 'twitch-de'
    line_dict = {'method_name': [], 'fpr': [], 'tpr': []}
    fig_path = dir_path + f'/{dataset_name}_roc.pdf'
    for condition in conditions:
        if condition['method_name'] in method:
            for para_path, paras_dict in zip(condition['paras_paths'], condition['paras_dict']):
                # client_num = int(paras_dict['federate.client_num'])
                data_name = paras_dict['data.type'].lower()
                if dataset_name in data_name:
                # if int(paras_dict['federate.client_num']) == client_num:
                    params = paras_dict.keys()
                    p = [s for s in params if param in s][0]
                    for p_v in param_v:
                        if int(paras_dict['federate.client_num']) == client_num and paras_dict[p] == p_v:
                            print(para_path)
                            results = []
                            round_dirs = [name for name in os.listdir(para_path) if
                                          os.path.isdir(os.path.join(para_path, name))]
                            for round_name in round_dirs:
                                round_dir = os.path.join(para_path, round_name)
                                if 'local' in condition['method_name']:
                                    if paras_dict['dataloader.batch_size'] == p_v:
                                        file_name = glob.glob(os.path.join(round_dir, '*.log'))
                                        with open(file_name[0], 'r') as f:
                                            log_content = f.read()
                                            # remove nan
                                            log_content = log_content.replace('nan', '0.5')
                                            m = ['test_roc_auc', 'test_acc', 'f1', 'test_recall', 'test_recall_macro',
                                                 'test_recall_weight', 'prec_5',
                                                 'prec_10', 'prec_20','fpr', 'tpr']
                                            # m = ['test_acc']
                                            finish, result = local_extrat(log_content, m)
                                    else:
                                        finish = False
                                else:  # fed results
                                    file_name = glob.glob(os.path.join(round_dir, 'eval_results_raw.log'))
                                    if file_name:
                                        with open(file_name[0], 'r') as f:
                                            log_content = f.read()
                                            log_content = log_content.replace('nan', '0.5')
                                            train_m = ['avg_loss', 'roc_auc', 'acc', 'f1', 'recall', 'recall_macro',
                                                       'recall_weight', 'prec_5', 'prec_10', 'prec_20', 'total', 'fpr',
                                                       'tpr']
                                            test_m = ['roc_auc', 'acc', 'f1', 'recall', 'recall_macro', 'recall_weight',
                                                      'prec_5', 'prec_10', 'prec_20', 'fpr', 'tpr']
                                            finish, result = fed_extrat(log_content, train_m, test_m, client_num)
                                    else:
                                        finish = False
                                if finish:
                                    results.append(result)
                                else:
                                    continue
                            if results:

                                    line_dict = comp_auc(results, condition['method_name'],line_dict)

                                    # result_dic = loc_comp_avg(results, m)
                                    # result_dic = {k: v for k, v in result_dic.items() if k in ['0','avg']}
                                # else:
                                #     result_dic = fed_comp_avg(results, test_m, client_num)
                                #     result_dic = result_dic[0]

    line_df = pd.DataFrame(line_dict)
    plot_auc(line_df,fig_path)
    print("test")







