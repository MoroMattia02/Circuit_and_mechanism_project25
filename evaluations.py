import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def quant_evaluation(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    print(f'Model Performances: ')
    print(f'Average f1: {f1}')
    print(f'Average precision: {precision}')
    print(f'Average recall: {recall}')
    print(f'Accuracy: {accuracy}')