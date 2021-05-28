import torch
import torch.nn
import torch.nn.functional as F

def logits_to_preds(logits):
    probs = F.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=1)

    return preds

def accuracy(preds, labels):
    return (preds == labels).float().mean()

def confusion_matrix(preds, labels, n_classes):

    conf_mat = torch.zeros(n_classes, n_classes)
    for i, j in zip(preds, labels):
        conf_mat[i, j] += 1

    return conf_mat

def precision(conf_mat):
    return torch.nan_to_num(torch.diagonal(conf_mat) / torch.sum(conf_mat, dim=1))

def recall(conf_mat):
    return torch.nan_to_num(torch.diagonal(conf_mat) / torch.sum(conf_mat, dim=0))

def f1(conf_mat):
    pre = precision(conf_mat)
    rec = recall(conf_mat)

    return torch.nan_to_num(2 * (pre * rec) / (pre + rec))

def logging_metrics(logits, labels):

    preds = logits_to_preds(logits)

    acc = accuracy(preds, labels)

    conf_mat = confusion_matrix(preds, labels, logits.size(-1))

    f1_macro = torch.mean(f1(conf_mat))

    return {'acc': acc.cpu().item(), 'f1': f1_macro.cpu().item()}
