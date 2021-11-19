import pandas as pd
import numpy as np


#Drop NaN columns with more than na_percent NaN on bads
def drop_nan(data, state_column_name, na_percent):
    
    data_bad = data.loc[data[state_column_name] == 1] 
    limit = len(data_bad) * na_percent
    data_bad = data_bad.dropna(thresh=limit, axis=1)
    list_cols = list(data_bad.columns)
    return data[list_cols]


def get_stat_df(data, data_x, state_column_name):
    
    data['GOOD'] = np.where(data[state_column_name] == 0, 1, 0)
    data['BAD'] = np.where(data[state_column_name] == 1, 1, 0) 
    
    first = True
    for column in data_x.columns:
        if first:
            stat_df = data.groupby([column])
            stat_df = pd.DataFrame({'COLUMN' : column, 
                                    'BAD' : stat_df['BAD'].apply(np.sum), 
                                    'GOOD' : stat_df['GOOD'].apply(np.sum)}).reset_index()
            stat_df = stat_df.rename(columns={column: "VARIABLE"})
            first = False
        else:
            to_concat =  data.groupby([column])
            to_concat = pd.DataFrame({'COLUMN' : column, 
                                      'BAD' : to_concat['BAD'].apply(np.sum), 
                                      'GOOD' : to_concat['GOOD'].apply(np.sum)}).reset_index()
            to_concat = to_concat.rename(columns={column: "VARIABLE"})
            stat_df = pd.concat([to_concat, stat_df])
            
    total_bad = np.sum(data['BAD'])
    total_good = np.sum(data['GOOD'])
    stat_df['BAD_PERCENT'] = stat_df['BAD'] / total_bad * 100
    stat_df['GOOD_PERCENT'] = stat_df['GOOD'] / total_good * 100
    stat_df['TOTAL_PERCENT'] = ((stat_df['BAD'] + stat_df['GOOD']) / (total_bad + total_good)) * 100
    
    return stat_df
