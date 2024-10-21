# train a dominant detector
from pygod.models import DOMINANT
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import torch
from pygod.generator import gen_contextual_outliers, gen_structural_outliers
from pygod.models import DOMINANT,CoLA,AnomalyDAE,AnomalyDAE
from pygod.models import ANEMONE
from pygod.metrics import eval_roc_auc
from collections import Counter
#data
data = Planetoid('./data', 'cora')[0]
data, ya,outlier_idx_c = gen_contextual_outliers(data = data, n=100, k=50)
data, ys,_ = gen_structural_outliers(data = data, m=10, n=10,outlierIdx=outlier_idx_c)
# test =torch.unique(torch.round(data.x[0]).long())
data.ay = torch.logical_or(ys, ya).int()
#model
model = ANEMONE(auc_test_rounds=100)
# model = CoLA(lr=0.003, subgraph_size=5,weight_decay=0,batch_size=100,epoch=100,negsamp_ratio=1,readout='weighted_sum',gpu=2,verbose=True) #AnomalyDAE()
model.fit(data)
#predict
prediction = model.predict(data)
# print('Labels:')
# print(labels)
auc_score = eval_roc_auc(data.ay.numpy(), prediction)
print('AUC Score0:', auc_score)
# # Dominant有用
outlier_scores = model.decision_function(data)
# # print('Raw scores:')
# # print(outlier_scores)
auc_score = eval_roc_auc(data.y.numpy(), outlier_scores)
print('AUC Score1:', auc_score)


#
# # the probability of the outlierness:
# prob = model.predict_proba(data)
# print('Probability:')
# print(prob)
# #To predict the labels with confidence:
# labels, confidence = model.predict(data, return_confidence=True)
# print('Labels:')
# print(labels)
# print('Confidence:')
# print(confidence)
#To evaluate the performance outlier detector:







