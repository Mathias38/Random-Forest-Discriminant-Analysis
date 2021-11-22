# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 09:59:52 2021

@author: mathias chastan
"""
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import _tree
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import graphviz
import copy

'''-----------------------------------------------------------------------------------------------------------------'''

class RandomForestDA(object):
    """
    Random Forest Discriminant Analysis can be use to find the cause of a problem in a large data set. Reponse variable should be 
    0 (GOOD) or 1 (BAD) and features should be the context data in dummies form : use pandas get_dummies(df, prefix_sep = ";") 
    
    :param  nb_trees:       Number of decision trees to use
    :param  max_depth:      Maximum depth of the trees
    :param cumulative_vip_treshold:     This treshold will determine where to split the operation dataframe and how much operation will be selected 
    (0.50 : select operation(s) that explain 50% of variance in current model)
    :param bad_accuracy_percent:    percentage bad accuracy of first model a model needs to be validated
    """
    def __init__(self, nb_trees = 100, depth=4, cumulative_vip_treshold = 0.5, bad_accuracy_percent = 0.9, max_features = "auto", vip_agregate_method = "sum"):
        self.nb_trees = nb_trees
        self.depth = depth
        self.cumulative_vip_treshold = cumulative_vip_treshold
        self.bad_accuracy_percent = bad_accuracy_percent
        self.max_features = max_features
        self.vip_agregate_method = vip_agregate_method
    
    """
    Creates the random forest model
    :param  x:  feautres dataset
    :param  y:  response variable
    """   
    def make_forest(self, x, y):
    
        rf = RandomForestClassifier(max_depth= self.depth, n_estimators = self.nb_trees, max_features = self.max_features)
        rf = rf.fit(x, y)
        return rf
    
    """
    creates vip df : dataframe containing variable importance for each column;variable    
    
    :param  forest:     the trained random forest object
    :param  x:  feature dataset
    """ 
    def get_vip(self, forest, x): 
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Print the feature ranking
        print("\n\n")
        print("Feature ranking:")
        
        columns = []
        vip = []
        for f in range(x.shape[1]):
            print(x.columns[indices[f]] +": "+ str(importances[indices[f]]))
            columns.append(x.columns[indices[f]])
            vip.append(importances[indices[f]])
          
        vip_df = pd.DataFrame({'COLUMN' :  columns, 'VIP' : vip})
        
        
        return vip_df
    
    """
    Discriminant analysis (see Mathias Chastan's PHD for details). Results are printed for the example but should be put in a dataframe for applicative usage.
    
    :param  data_x:     features dataset
    :param  prep_data:  features dataset in dummies structure (see panda dummy functon)
    :param  y:  response variable
    """ 
    def discriminant_analysis(self, data_x, prep_data, y):

        rf = self.make_forest(prep_data, y)
        preds = rf.predict(prep_data)
        preds = list(preds)
        vip = self.get_vip(rf, prep_data)
                
        #Three possible methods here (sum,max,mean) : sum is mathematicly more suited
        
        if self.vip_agregate_method == "sum":
        
            vip_col = self.get_vip_by_column_sum(vip)
            
        elif self.vip_agregate_method == "mean":
            
            vip_col = self.get_vip_by_column_mean(vip)
            
        elif self.elf.vip_agregate_method == "max":
            
            vip_col = self.get_vip_by_column_max(vip)
            
        else :
            print("ERROR: "+ self.vip_agregate_method+" is not a supported agregation method")
        
        cols = vip_col['COLUMN']
        last_ok_bad_acc_cols = vip_col['COLUMN']
        treshold = self.cumulative_vip_treshold
        min_bad_acc = self.bad_accuracy(y , preds) * self.bad_accuracy_percent
        
        #This loop will make model with all variables and then drop a part of the operations depending on cumulative vip (explained variance)
        #The loop stops when the bad accuracy treshold is reached or when there is only one operation left
        #Variables are divided between context for each operation (can be equipment / chamber / recipe)
        while len(cols) > 1:
                        
            vip_col_sorted = vip_col.sort_values(by=['VIP_SUM'], ascending = False).reset_index()
            vip_col_sorted = vip_col_sorted.drop(['index'], axis = 1)
            vip_col_sorted['CUMUL_VIP_SUM'] =  vip_col_sorted['VIP_SUM'].cumsum()
            
            #Discard columns that are under the cumulated variable importance under treshold
            for i in range(0,len(vip_col_sorted)):
                
                if vip_col_sorted.iloc[i]['CUMUL_VIP_SUM'] > treshold:
                    idx = i
                    break   
                    
            vip_col_sorted = vip_col_sorted[:(idx + 1)]
            old_cols = copy.copy(cols)
            cols = vip_col_sorted['COLUMN']
            
            #If no columns are discarded reduce the treshold (you can change the 0.05 step to descend faster or slower)
            while len(old_cols) <= len(cols):
                treshold = treshold - 0.05
                print("treshold")
                print(treshold)
                print("old_cols")
                print(old_cols)
                print("cols")
                print(cols)
                for i in range(0,len(vip_col_sorted)):
                
                    if vip_col_sorted.iloc[i]['CUMUL_VIP_SUM'] > treshold:
                        idx = i
                        break   
                    
                vip_col_sorted = vip_col_sorted[:(idx + 1)]
                old_cols = copy.copy(cols)
                cols = vip_col_sorted['COLUMN']
                    
            new_data = pd.get_dummies(data_x[cols], prefix_sep = ";")
               
            #Drop missing columns
            for col in list(new_data.columns):
                split = col.split(";")
                if split[1] == "MISSING":
                    new_data = new_data.drop([col], axis = 1) 
                  
            rf = self.make_forest(new_data, y)
            preds = rf.predict(new_data)
            preds = list(preds)
            vip = self.get_vip(rf, new_data)
                        
            if self.vip_agregate_method == "sum":
            
                vip_col = self.get_vip_by_column_sum(vip)
            
            elif self.vip_agregate_method == "mean":
            
                vip_col = self.get_vip_by_column_mean(vip)
            
            elif self.elf.vip_agregate_method == "max":
            
                vip_col = self.get_vip_by_column_max(vip)
                    
            self.print_model_metrics(y, preds)
            print("\n")
                       
            #Save the last columns for which model bad accuracy is greater than minimum bad accuracy
            if self.bad_accuracy(y, preds) >= min_bad_acc:
                last_ok_bad_acc_cols = copy.copy(cols) 
                              
            print("min bad accuracy")
            print(min_bad_acc)
            
        #If the 1 column model has a bad accuracy under minimum bad accuracy go back to the last ok model
        if self.bad_accuracy(y, preds) < min_bad_acc:
            new_data = pd.get_dummies(data_x[last_ok_bad_acc_cols], prefix_sep = ";")
                               
            #Drop missing machines
            for col in list(new_data.columns):
                
                split = col.split(";")
                if split[1] == "MISSING":
                    new_data = new_data.drop([col], axis = 1)
                               
            rf = self.make_forest(new_data, y)
            preds = rf.predict(new_data)
            preds = list(preds)
            vip = self.get_vip(rf, new_data)
                        
            if self.vip_agregate_method == "sum":
            
                vip_col = self.get_vip_by_column_sum(vip)
            
            elif self.vip_agregate_method == "mean":
            
                vip_col = self.get_vip_by_column_mean(vip)
            
            elif self.elf.vip_agregate_method == "max":
            
                vip_col = self.get_vip_by_column_max(vip)
                
            self.print_model_metrics(y, preds)        
            print("min bad accuracy")
            print(min_bad_acc)
    
    """
    Creates a dataframe sumarizing the random forest in a comprehensible way
    
    :param  rf:     trained random forest object
    :param  data:  trimmed dataset : only
    """ 
    def get_easy_forest_table(rf, data):
        
        ftt = rf.forest_to_table(rf, data.columns)
        ftt["good_percent"] = ftt["Good"] / (ftt["Good"] + ftt["Bad"])
        ftt["bad_percent"] = ftt["Bad"] / (ftt["Good"] + ftt["Bad"])
        ftt["1"] = "des individus évalués par" 
        ftt["2"] = ftt["Columns"]
        ftt["3"] = "sont mauvais"
        ez_forest = ftt.loc[ftt['Values'] == 1] 
        ez_forest = ez_forest[["0","1","2","3"]]
        return ez_forest

    """
    Creates a graph object of the tree with graphviz.
    
    :param  t:  trained tree object
    :param  x:  features dataset
    :param class_names: y possible classes
    """             
    def draw_tree(self, t, x, class_names):
        dot_data = tree.export_graphviz(t, out_file=None, 
                                        feature_names=x.columns,  
                                        class_names = class_names,
                                        filled=True)
    
        # Draw graph
        graph = graphviz.Source(dot_data, format="png") 
        return graph
    
    """
    Creates a dataframe sumarzing a tree.
    
    :param  t:  trained tree object
    :param  feature_names:  column names of the features dataset
    """   
    def tree_to_table(self, t, feature_names):
        
        columns = []
        signs = []
        tresholds = []
        results = []
        sequences = []
        texts = []
        
        tree_ = t.tree_
        feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
        
        def recurse(node, depth, i):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                columns.append(name)
                signs.append("<=")
                tresholds.append(threshold)
                sequences.append(i)
                if i != 0:
                    results.append(tree_.value[node])
                texts.append("if")
                recurse(tree_.children_left[node], depth + 1, i + 1)
                columns.append(name)
                signs.append("<=")
                tresholds.append(threshold)
                sequences.append(i)
                texts.append("elseif")            
                recurse(tree_.children_right[node], depth + 1, i + 1)
            else:
                results.append(tree_.value[node])
                i = i + 1
        

        recurse(0, 1, 0)
    

            
        good = self.translate_results(results)['good']
        bad =  self.translate_results(results)['bad']
        names =  self.translate_tresholds(tresholds, signs, texts)
          
        if not texts :
            texts.append("UNE SEULE FEATURE")  
            sequences.append("na")
            columns.append("na")
            names.append("na")
            
        df = pd.DataFrame({'Texts' :  texts, 
                                       'Sequences' : sequences, 
                                       'Columns' : columns, 
                                       'Values' : names, 
                                       'Good' : good, 
                                       'Bad' : bad }).reset_index() 
        return df
    
    """
    This function is used by the tree_to_table function
    """       
    def translate_results(self, results):
        
        good = []
        bad = []
        for i in range(0,len(results)):
           if results[i][0] is not None :
               good.append(results[i][0][0])
               bad.append(results[i][0][1])
           else:
               good.append(None)
               bad.append(None)
         
        d = dict();  
        d['good'] = good 
        d['bad'] = bad
        return d
    
    """
    This function is used by the tree_to_table function
    """      
    def translate_tresholds(self, tresholds, signs, text):   
        names = []
        for i in range(0, len(tresholds)):
            if tresholds[i] is not None :
                if text[i] == "if" :
                    if signs[i] == "<=" :
                        names.append(int(tresholds[i] - 0.5))
                    if signs[i] == ">" :
                        names.append(int(tresholds[i] + 0.5))
                if text[i] == "elseif" :
                    if signs[i] == "<=" :
                        names.append(int(tresholds[i] + 0.5))
                    if signs[i] == ">" :
                        names.append(int(tresholds[i] - 0.5))                
            else:
                names.append(None)
    
        return names
    
    """
    print the model metrics. Used by discriminant analysis function to monitor the models quality.
    
    :param  y_true:  y real value from the test data set
    :param  y_preds:  y predictions for the test data set
    """ 
    def print_model_metrics(self, y_true,y_preds):
        print("CONFUSION MATRIX \n")
        print(confusion_matrix(y_true,y_preds))
        print(" \n\nACCURACY SCORE \n")
        print(accuracy_score(y_true,y_preds))
        print("\n\n BAD ACCURACY SCORE \n")
        print(self.bad_accuracy(y_true,y_preds))
    
    """
    Calculates and returns prediction accuracy for the "bad" class
    
    :param  y:  y real value from the test data set
    :param  y_preds:  y predictions for the test data set
    """    
    def bad_accuracy(self, y, y_preds):
        
        y_true = y.reset_index()['CLASS']
        total_bads = 0
        good_preds = 0
        for i in range(0, len(y_true)):
            if y_true[i] == 1:
                total_bads = total_bads + 1
                if y_true[i] == y_preds[i]:
                    good_preds = good_preds + 1
                    
        return good_preds / total_bads
    
    """
    Creates a dataframe sumarzing a forest.
    
    :param  forest:  trained random forest object
    :param  y_preds:  y predictions 
    """        
    def forest_to_table(self, forest, feature_names):    
        first = True
        for t in forest.estimators_:
            if first:
                forest_df = self.tree_to_table(t, feature_names)
                first = False
            else:
                forest_df = pd.concat([forest_df, self.tree_to_table(t, feature_names)])
            
           
        res_df = forest_df.groupby(['Columns','Values'])
        res_df = pd.DataFrame({'Good' : res_df['Good'].apply(np.sum), 'Bad' : res_df['Bad'].apply(np.sum)}).reset_index()
    
        return res_df
    
    
    
    """    
    Gets the column string from a column;dummy_variable string

    :param  col_x:  column;dummy_variable string which is constructed by using pandas get_dummies function
    """
    def get_column_from_column_dummy(self, col_x):
        split = col_x.split(";")
        return split[0]
    
    """    
    Calculate and return vip dataframe : variable importance by columns (sum by column)

    :param  vip:    vip by column;variable 
    """
    def get_vip_by_column_sum(self, vip):

        vip['COLUMN'] = vip.apply(lambda row: self.get_column_from_column_dummy(row['COLUMN']),axis=1)
        vip_gp = vip.groupby(['COLUMN'])
        vip_gp = pd.DataFrame({'VIP_SUM' : vip_gp['VIP'].apply(np.sum)}).reset_index()

        return vip_gp
    
    """    
    Calculate and return vip dataframe : variable importance by columns (mean by column)

    :param  vip:    vip by column;variable 
    """
    def get_vip_by_column_mean(self, vip):

        vip['COLUMN'] = vip.apply(lambda row: self.get_column_from_column_dummy(row['COLUMN']),axis=1)
        vip_gp = vip.groupby(['COLUMN'])
        vip_gp = pd.DataFrame({'VIP_SUM' : vip_gp['VIP'].apply(np.sum)}).reset_index()

        return vip_gp

    """    
    Calculate and return vip dataframe : variable importance by columns (max by column)

    :param  vip:    vip by column;variable 
    """
    def get_vip_by_column_max(self, vip):

        vip['COLUMN'] = vip.apply(lambda row: self.get_column_from_column_dummy(row['COLUMN']),axis=1)
        vip_gp = vip.groupby(['COLUMN'])
        vip_gp = pd.DataFrame({'VIP_SUM' : vip_gp['VIP'].apply(np.amax)}).reset_index()

        return vip_gp
    
  

    

'''-----------------------------------------------------------------------------------------------------------------'''
