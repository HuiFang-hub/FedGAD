# -*- coding: utf-8 -*-
# @Time    : 26/03/2023 17:42
# @Function:
from src.dgld.models import *
def DGLDmodel(model_config):
    if model_config.type in ['DOMINANT','AnomalyDAE','ComGA','DONE','AdONE','CONAD','ALARM','ONE','GAAN','GUIDE','CoLA',
                        'AAGNN', 'SLGAD','ANEMONE','GCNAE','MLPAE','SCAN']:
        model = eval(f'{model_config.type}(**args_dict["model"])')
    else:
        raise ValueError(f"{model_config.type} is not implemented!")
    return model