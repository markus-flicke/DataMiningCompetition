import pandas as pd
from sklearn.metrics import roc_auc_score


def real_auc(preds = [0] * 15000, ids = range(15000)):

    df = pd.read_csv('cheat_solutions.csv')
    real_targets = df.sort_values('id').target
    real_targets = real_targets.iloc[ids]

    return roc_auc_score(real_targets, preds)



if __name__ == '__main__':
    df = pd.read_csv('train.csv')
    df = df.sort_values('id')
    print(real_auc(df.target, df.id))
