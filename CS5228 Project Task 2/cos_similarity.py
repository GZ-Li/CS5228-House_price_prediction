from scipy import spatial
import numpy as np
import pandas as pd

feature_weight = [0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1]

def normalize(df_sample):
    return (df_sample-df_sample.mean())/df_sample.std()

def similarity(row_indicator, df_sample, k):

    # Normalize data before calculating difference
    row = row_indicator
    df_normalize = normalize(df_sample)

    if not isinstance(row_indicator, pd.DataFrame):
        row = df_normalize.iloc[row_indicator]
        row = pd.DataFrame([row])

    row = row.iloc[0]

    results = []
    for index, train in df_normalize.iterrows():
        results.append(spatial.distance.cosine(row, train, feature_weight))
    
    vals = np.array(results)
    sort_index = np.argsort(vals)
    df_recommend = df_sample.iloc[sort_index[1:k+1], :]
    df_recommend = df_recommend

    return df_recommend