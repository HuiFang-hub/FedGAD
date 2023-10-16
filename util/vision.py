# -*- coding: utf-8 -*-
# @Time    : 17/04/2023 16:31
# @Function:
import matplotlib.pyplot as plt
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from string import ascii_letters
import numpy as np
from collections import Counter
def multi_plot(data_dict):
    for k, v in data_dict.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.show()

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



def plot_violin(x, y, data, title):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    ax = sns.violinplot(x=x, y=y, data=data)
    ax.set_title(title, fontsize=16)
    plt.show()


def plot_roc(fpr, tpr, roc_auc):
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def plot_violinplot(violin_df):
    sns.set_theme(style="whitegrid")
    sns.violinplot(data=violin_df, x="client_id", y="y_label", hue="anomaly_label",
                   split=True, inner="quart", linewidth=1,
                   palette={1: "b", 0: ".85"})
    legend_labels = {0: "Non-anomaly", 1: "Anomaly"}
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles, legend_labels.values())
    sns.despine(left=True)
    plt.show()


def plot_auc(line_df, fig_path):
    # mpl.rcParams['text.usetex'] = True
    # mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
    sns.set_theme(style="darkgrid")
    hue_order = ['$FedGAD_{-GNR}$', '$FedGAD$', '$loc_1$', '$loc_2$', '$loc_3$', 'Global']
    line_colors = ['#FFD0E9', '#B9191A', '#DBE7CA', '#99BADF', '#99CDCE', '#999ACD']
    # line_colors = ['#A95465', '#A97C81','#A6DADB', '#FAB378', '#137355', '#5E5851', ]
    color_palette = dict(zip(hue_order, line_colors))
    ax = sns.lineplot(x="fpr", y="tpr",err_style = "band", hue="method_name", hue_order=hue_order, data=line_df, palette=color_palette)
    # ax = sns.lineplot(x="fpr", y="tpr", hue="method_name", hue_order=hue_order, data=line_df)

    lines = ax.get_lines()
    for line in lines:
        if line.get_label() == "FedGAD":
            line.set_linewidth(5.5)

    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random guess')
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.subplots_adjust(right=0.7, top=0.7)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.show()
    # print("test")


#
# def draw(file_paths,dir,name):
#     scoremarkers=["v","s","*","o","x","+"]
#     # accmarkers=["v","s","*","o","x"]
#     for i, path in enumerate(file_paths):
#         fmri=pd.read_csv(path,sep=',',) #header=None,names=["score"],index_col=False
#         num=sum(1 for line in open(path))
#         # fmri["score"]=fmri["score"]*100
#         # sns.barplot(x="alpha", y="RS", data=fmri)
#         # sns.barplot(x="alpha", y="CS", data=tips)
#         # sns.barplot(x="alpha", y="SP", data=tips)
#         a=0
#         ax=sns.lineplot(x="SNPC",y="RS",err_style = "band",ci="sd",marker=scoremarkers[a],linewidth=3,
#             # hue="region", style="event",
#             data=fmri)
#         a=a+1
#         ax=sns.lineplot(x="SNPC",y="CS",err_style = "band",ci="sd",marker=scoremarkers[a],linewidth=3,
#             # hue="region", style="event",
#             data=fmri)
#         a=a+1
#         ax=sns.lineplot(x="SNPC",y="SP",err_style = "band",ci="sd",marker=scoremarkers[a],linewidth=3,
#             # hue="region", style="event",
#             data=fmri)
#         plt.xlabel("SNPC",fontsize=20)
#         plt.ylabel('Accuracy,%',fontsize=20)
#         plt.yticks(np.arange(55, 72, 5))
#         plt.legend([r"S $\rightarrow$ P",r"C $\rightarrow$ S", r"R $\rightarrow$ S"],loc="lower left",fontsize=12)
#         plt.savefig(os.path.join(dir+ name), format="pdf",bbox_inches="tight",dpi = 400)
#         # plt.clf()

def heat_map(df,fig_path):
    sns.set_theme(style="white")
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    maxv = df.max().max()
    minv =df.min().min()
    center = df.stack().median()
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(df, cmap=cmap,vmin=minv, vmax= maxv, center=center,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.xlabel('$\lambda_0$', fontsize=20)
    plt.ylabel('$\lambda_1$', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.show()


def rectangle_plot(df, metric_name, fig_path):
    x = np.arange(len(df.index))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    plt.rcParams['font.size'] = 20
    line_colors = ['#D3E2B7','#CFAFD4']
    for attribute, measurement in df.iteritems():
        offset = width * multiplier
        rects = plt.bar(x + offset, measurement, width, label=attribute, color=line_colors[multiplier])
        # plt.bar_label(rects)
        # plt.plot([], [], color=line_colors[multiplier], label=f'Line {multiplier+1}')  # 添加一个空的plot，用于生成线条的图例
        multiplier += 1

    plt.xlabel('$\lambda_\\alpha $', fontsize=20)
    if metric_name=='roc_auc':
        metric_name='AUC'
    else:
        metric_name = metric_name.upper()
    plt.ylabel(f'{metric_name}', fontsize=20)
    plt.xticks(x, df.index, fontsize=20)
    plt.yticks(fontsize=20)


    max_value = df.max().max()
    min_value = df.min().min()
    y_range = max_value - min_value
    padding = 0.1 * y_range
    plt.ylim(min_value - padding, max_value + padding)

    plt.legend()

    plt.box(False)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.show()


if __name__ == '__main__':
    alpha_level = ["0.0", "0.2", "0.4", "0.8", "1.0"]
    penguin_means = {
        'twitch_pt': (18.35, 18.43, 14.98, 18.43, 14.98),
        'amazon': (38.79, 48.83, 47.50, 48.83, 47.50),
    }
    df = pd.DataFrame(penguin_means, index=alpha_level)
    rectangle_plot(df, 'auc','')


    # Generate a large random dataset
    # rs = np.random.RandomState(33)
    # d = pd.DataFrame(data=rs.normal(size=(100, 26)),
    #                  columns=list(ascii_letters[26:]))
    # heat_map(d)



    # path = '../exp/fedsageGod_cola_on_cora-inj-before_lr0.00155_lstep4/'
    # # file_name = 'exp_print.log'
    # file_name = 'eval_results_raw.log'
    # with open(path+file_name, 'r') as f:
    #     log_text = f.read()
    #
    # role ='Client'
    # metric = ['train_avg_loss','train_roc_auc','train_acc','train_recall']
    #
    # # role = 'Serve'
    # # metric = ['test_roc_auc']
    #
    # result = role_result_match(role, log_text, metric )
    # # print(new_dict)
    # if role == "Client":
    #     for k,v in result.items():
    #         new_dict = result[k]
    #         multi_plot(new_dict)
    # else:
    #     multi_plot(result)
    # 定义数据


    # sns.set_theme(style="whitegrid")
    #
    # 生成随机数据
    # np.random.seed(0)
    # client_ids = np.random.randint(0, 1, size=20)
    # y_labels = np.random.randint(0, 8, size=20)
    # anomaly_labels = np.random.randint(0, 2, size=20)
    #
    # # 创建 DataFrame
    # data = {'client_id': client_ids, 'y_label': y_labels, 'anomaly_label': anomaly_labels}
    # df = pd.DataFrame(data)
    # print(df)
    #
    # # 输出每列的值的个数
    # client_id_counts = Counter(df['client_id'])
    # y_label_counts = Counter(df['y_label'])
    # anomaly_label_counts = Counter(df['anomaly_label'])
    #
    # # 打印每列的值的个数
    # print("client_id counts:", client_id_counts)
    # print("y_label counts:", y_label_counts)
    # print("anomaly_label counts:", anomaly_label_counts)
    # plot_violinplot(df)
    # # 调用函数绘制小提琴图
    # plot_violin(x='Group', y='Value', data=data, title='Violin Plot')
    # sns.violinplot(x='Group', y='Value', data=data,split=True, inner="quart", linewidth=1)
    # sns.despine(left=True)
    # plt.show()
    #
    # import seaborn as sns
    # import random
    # sns.set_theme(style="whitegrid")
    #
    # # Load the example tips dataset
    # tips = sns.load_dataset("tips")
    # tips['label'] = [random.randint(1, 7) for _ in range(len(tips['total_bill']))]
    # # Draw a nested violinplot and split the violins for easier comparison
    # sns.violinplot(data=tips, x="day", y="label", hue="smoker",
    #                split=True, inner="quart", linewidth=1,
    #                palette={"Yes": "b", "No": ".85"})
    # sns.despine(left=True)
    # plt.show()
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    #
    # sns.set_theme(style="darkgrid")
    #
    # # Load an example dataset with long-form data
    # fmri = sns.load_dataset("fmri")
    #
    # # Plot the responses without grouping
    # sns.lineplot(x="timepoint", y="signal",
    #              hue="region",
    #              data=fmri)
    # plt.xlabel('$FedGAD_{-GNR}$', fontsize=20)
    # legend_labels = {'parietal': "$FedGAD_{-GNR}$", 'frontal': "$FedGAD_{0}$"}
    # plt.tight_layout()
    # plt.show()