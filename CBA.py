#!/usr/bin/env python
# coding: utf-8

# In[549]:


import pandas as pd
import numpy as np
from frozendict import frozendict
import random


# In[452]:


# WIP
class Rule:
    
    def __init__(self, itemset, class_count, len_D):
        self.itemset = itemset
        self.class_count = class_count
        
        if len(set(class_count.values()))==1:
            self.result = random.choice(class_count.keys())
        else:
            self.result = max(class_count, key=lambda x: class_count[x])
        
        self.conf = float(class_count[self.result])/sum(class_count.values())
        self.sup = class_count[self.result] / float(len_D)


# In[453]:


# WIP
class Rules:
    
    def __init__(self):
        self.rules = {}
        
    
    def get_rule(self, key):
        length = len(key)
        
        if length not in self.rules:
            return None
        else:
            return self.rules[length].get(key,None)
        
    def get_rules_by_length(self, length):
        if length not in self.rules:
            return []
        else:
            return self.rules[length].values()
    
    def get_itemset_by_length(self, length):
        if length not in self.rules:
            return set()
        else:
            return set(self.rules[length].keys())
    
    def get_itemset_rules_by_length(self, length):
        if length not in self.rules:
            return {}
        else:
            return self.rules[length]
    
    def add(self, rule):
        if rule==None:
            return
        
        length = len(rule.itemset)
        
        if length not in self.rules:
            self.rules[length] = rule
        else:
            self.rules[length][rule.itemset] = rule
    
    def remove(self, itemset):
        length = len(itemset)
        
        self.rules[length].pop(itemset)


# In[558]:


class Classifier:
    
    def __init__(self, rule_builder):
        self.rule_builder = rule_builder
        self.rules = None
        self.sorted_CARS = None
        
    def sort_rules(self, len_D):
        sorted_CARS = []
        for values in self.rule_builder.CARS.values():
            sorted_CARS.extend(list(values.items()))
        self.sorted_CARS = sorted(sorted_CARS, key=lambda x: (x[1][1][x[1][0]] / sum(x[1][1].values()), x[1][1][x[1][0]]/len_D, len(x[0])), reverse=True)
    
    def build_classifier(self, df, target_col):
        len_D = len(df)
        self.sort_rules(len_D)
        temp_df = df
        rules = []
        for CARS in self.sorted_CARS:
            cond, result = CARS
            cond_df = temp_df.loc[(temp_df[list(cond)] == pd.Series(cond)).all(axis=1)]
            correct = cond_df[cond_df[target_col]==result[0]]

            if len(correct)!=0:
                temp_df = temp_df.drop(index=cond_df.index) 
                if len(temp_df)==0:
                    default_class = random.choice(df[target_col].unique())
                else:
                    default_class = temp_df[target_col].value_counts().idxmax()
                total_error = (len(cond_df) - len(correct)) + len(temp_df[temp_df[target_col]!=default_class])
                error = {'default': len(temp_df[temp_df[target_col]!=default_class]), 'class':(len(cond_df) - len(correct))}
                rules.append([CARS, default_class, total_error, error])

        lowest_error_id = np.argmin([x[2] for x in rules])
        pruned_rules = rules[:lowest_error_id+1]
        self.rules = pruned_rules
        
    def predict(self, df):
        temp_df = df.copy()
        ans = df.copy()
        # Setting all to default class
        ans['prediction'] = rules[-1][1]
        for rule in self.rules:
            cond, default_class, _, _ = rule
            cond, prediction = cond
            # Filtering rows that fulfil rule condition
            cond_df = temp_df.loc[(temp_df[list(cond)] == pd.Series(cond)).all(axis=1)]
            # Setting prediction
            ans.loc[cond_df.index, 'prediction'] = prediction[0]
            # Removing rows that has been predicted
            temp_df = temp_df.drop(index=cond_df.index)

        return ans
        


# In[537]:


class RuleGenerator:
    def __init__(self, min_sup=0.2, min_conf=0.6, weighted=False):
        self.min_sup = min_sup
        self.min_conf = min_conf
        self.CARS = None
        self.weighted = weighted
    
    def generate_rules(self, df, target_col):
        len_D = len(df)
        class_labels = df[target_col].unique()
        variables = list(df.columns)
        variables.remove(target_col)
        k=1
        F_1 = {}
        for variable in variables:
            temp = df.groupby(by=[variable, target_col]).size().unstack(level=1).reset_index()
            temp = temp.fillna(0)
        #     print(temp)
            temp.apply(lambda x: F_1.update(self.convert_to_rule_items(x, [variable], class_labels, len_D)), axis=1)

        self.CARS = {}
        self.CARS[1] = self.gen_rules(F_1, len_D)

        F = {}
        F[1] = F_1
        # print(F_1.keys())
        frequent_variables = set([x for x in F_1.keys()])
        # print(frequent_variables)
        while len(F[k])!=0:
            k+=1
            # Includes generating candidate itemset and finding frequent itemset
            F[k] = self.gen_itemset(F[k-1], df, target_col, frequent_variables)
            frequent_variables = set([x for x in F[k].keys()])

            # Build rules from frequent itemset
            self.CARS[k] = self.gen_rules(F[k], len_D)
            # Pruning rules
            self.CARS[k] = self.prune_rules(self.CARS[k], self.CARS[k-1])
        
    def convert_to_rule_items(self, x, variables, class_label, len_D):
        condset = {}
        for variable in variables:
            condset[variable] = x[variable]

        condset = frozendict(condset)
        class_count = {}
        # When label don't exist in group
        for label in class_label:
            if label not in x:
                class_count[label] = 0
            else:
                class_count[label] = x[label]
        major_class = max(class_count, key=lambda x: class_count[x])

        # Removing non frequent itemset
#         if (class_count[major_class]/float(len_D))<self.min_sup:
#             return {}

        if self.weighted:
            for label in class_label:
                if (class_count[label] / float(len_D)) >= self.min_sup[label]:
                    return {condset: (major_class, class_count)}
        else:
            if (class_count[major_class]/float(len_D))>=self.min_sup:
                return {condset: (major_class, class_count)}

        return {}

    def gen_rules(self, F_k, len_D):

        rules = {}

        for condset, (major_class, class_count) in F_k.items():

            conf = class_count[major_class] / float(sum(class_count.values()))

            if conf>self.min_conf:
                # Checking if the support are the same for all classes
                if set(class_count.values())==1:
                    # Choose a random class
                    major_class = random.choice(class_count.keys())

                rules[condset] = (major_class, class_count)

        return rules
    
    
    def gen_itemset(self, F, df, target_col, itemsets):
        F_new = {}
        itemsets = set(itemsets)

        condset = list(F.keys())

        for i in range(len(condset)):
            cond = condset[i]
            temp = df.copy()
            cols = []
            for item in cond.items():
                value = item[1]
                col = item[0]
                temp = temp[temp[col]==value]
                cols.append(col)

            for j in range(i+1, len(condset)):
                itemset = condset[j]
                #Line 20 to 25 is the Apriori principle, where we merge 2 frequent superset into a candidate key
                # Checking if 2 itemsets differ only by 2 conditions
                itemset_keys = itemset.keys()
                cond_keys = cond.keys()
                if len(set(itemset_keys) - set(cond_keys))==1 and len(set(itemset.items())^set(cond.items()))==2:
                    variable = (set(itemset_keys) - set(cond_keys)).pop()
                    value = itemset.get(variable)

                    # Candidate generation
                    temp_2 = temp[temp[variable]==value]
                    groupby = cols+[variable, target_col]

                    # Calculating frequency
                    # TODO: Can be optimize further as we already filtered out the candidate, hence only need groupby class but this will affect convert_to_rule_items
                    group = temp_2.groupby(by=groupby).size().unstack(level=-1).reset_index()
                    group = group.fillna(0)
                    # Converting candidates in frequent itemset
                    group.apply(lambda x: F_new.update(convert_to_rule_items(x, groupby[:-1], class_labels, min_sup, len_D)), axis=1)


        return F_new
    
    
    
    def prune_rules(self, r, r_):
        # r and r_ is a python dict
        to_remove = []
        for superset in r.keys():
            # superset is a frozendict
            superset_set = set(superset.items())
            superset_value = r[superset]

            for subset in r_.keys():
                # subset is a frozendict
                subset_set = set(subset.items())

                if subset_set<superset_set:
                    subset_value = r_[subset]

                    superset_class = superset_value[0]
                    superset_error = superset_value[1][superset_class] / sum(superset_value[1].values())

                    subset_class = subset_value[0]
                    subset_error = subset_value[1][subset_class] / sum(subset_value[1].values())

                    if superset_error>=subset_error:
                        to_remove.append(superset)
                        # break to as the superset has already been removed, no further testing is needed
                        break

        for key in to_remove:
            r.pop(key)
        return r

