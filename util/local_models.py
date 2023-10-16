from src.pygod.models import ANEMONE,DOMINANT, CoLA, AnomalyDAE
from src.federatedscope.gfl.fedAnemone.metrics import *
from src.pygod.metrics import eval_roc_auc
import logging
import matplotlib.pyplot as plt
# logger = logging.getLogger(__name__)
from util.vision import plot_roc
from collections import Counter
root_logger = logging.getLogger("src.federatedscope")

def local(data, cfg):
    # root_logger.info("test")
    res = {}
    anomaly_labels = []

    for i, graphset in data.items():
        model = None
        subdata = graphset['data']
        test_data = graphset['test_data']
        root_logger.info(f"client:{i},anomaly label proportion:{Counter(subdata.ay.tolist())}")
        anomaly_labels.append(subdata.ay)
        if cfg.model.type.lower() == "cola":
            model = CoLA(lr=cfg.train.optimizer.lr, weight_decay=0, subgraph_size=4,
                         batch_size=cfg.dataloader.batch_size,
                         epoch=cfg.federate.total_round_num,
                         negsamp_ratio=1, gpu=cfg.device, verbose=True,client=i)  # AnomalyDAE()
            model.fit(subdata, y_true=subdata.ay)
        elif cfg.model.type.lower() == "anemone":
            model = ANEMONE(lr=cfg.train.optimizer.lr, gpu=cfg.device, subgraph_size=4,verbose=True,
                             batch_size=cfg.dataloader.batch_size,epoch=cfg.federate.total_round_num,
                             alpha=cfg.model.alpha,negsamp_ratio_patch=cfg.model.negsamp_ratio_patch,
                             negsamp_ratio_context=cfg.model.negsamp_ratio_context,client=i,
                            auc_test_rounds = cfg.model.test_rounds)
            model.fit(subdata)
        # predict
        # prediction,pred_score = model.predict(test_data)
        # print('Labels:')
        # print(labels)
        # auc_score0,fpr, tpr,prec_5,prec_10,prec_20 = result_auc(test_data.ay.numpy(), pred_score)
        # print('AUC Score0:', auc_score0)

        pred_score = model.decision_function(test_data)
        # model._process_decision_scores()
        prediction = model.labels_
        roc_auc,fpr, tpr,prec_5,prec_10,prec_20 = result_auc(test_data.ay, pred_score)
        # plot_roc(fpr, tpr, roc_auc)
        acc, recall,f1 = result_acc_rec_f1(test_data.ay, prediction)
        recall_macro,recall_weight = calculate_macro_recall(test_data.ay, prediction)
        res[i] = {'test_roc_auc': roc_auc, 'test_acc': acc, 'f1':f1, 'test_recall': recall,
                  'test_recall_macro':recall_macro,'test_recall_weight':recall_weight,
                  'prec_5':prec_5,'prec_10': prec_10,'prec_20':prec_20,
                  'fpr':fpr,'tpr':tpr}
        print("AUC {:.5f}".format(roc_auc), end='')
        print(" | ACC {:.5f}".format(acc), end='')
        print(" | F1 {:.5f}".format(f1), end='')
        print(" | Recall {:.5f}".format(recall), end='')
        print(" | Recall_Macro {:.5f}".format(recall_macro), end='')
        print(" | Recall_Weight {:.5f}".format(recall_weight), end='')
        print(" | Precision_5 {:.5f}".format(prec_5), end='')
        print(" | Precision_10 {:.5f}".format(prec_10), end='')
        print(" | Precision_20 {:.5f}".format(prec_20), end='')
        print()

    root_logger.info(f"results:{res}")
    # print(anomaly_labels)

