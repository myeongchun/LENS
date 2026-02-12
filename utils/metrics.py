from sklearn.metrics import roc_auc_score, average_precision_score

def calculate_metrics(y_true, y_score):
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    
    return {
        'auroc': auroc,
        'aucpr': auprc
    }