# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 09:52:51 2021

@author: mathias chastan
"""

import pandas as pd
from source.RandomForestDA import RandomForestDA

'''-----------------------------------------------------DATA IMPORT---------------------------------------------------'''

data = pd.read_csv('data/example_data.csv', sep=';')

y = data['CLASS']
data_x = data.drop(['CLASS', 'CAFETIERE_ID'], axis = 1)
data_x = data_x.fillna('MISSING')

#Change to categorical data frame
prep_data = pd.get_dummies(data_x, prefix_sep = ";")

#Drop missing columns
for col in list(prep_data.columns):
    split = col.split(";")
    if split[len(split)-1] == "MISSING":
        prep_data = prep_data.drop([col], axis = 1)

'''-------------------------------------------------------------------------------------------------------------------'''

'''--------------------------------------------------DISCRIMINANT ANALYSIS---------------------------------------------'''

rfda = RandomForestDA()
#discriminant analysis prints the results for test purpose (change the return in the function for applicative usage)
rfda.discriminant_analysis(data_x, prep_data, y)



