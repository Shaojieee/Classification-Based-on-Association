import pandas as pd
import numpy as np
from frozendict import frozendict
import random

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
        k = 1
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
        while len(F[k]) != 0:
            # print(k)
            k += 1
            # Includes generating candidate itemset and finding frequent itemset
            F[k] = self.gen_itemset(F[k - 1], df, target_col, frequent_variables)
            frequent_variables = set([x for x in F[k].keys()])

            # Build rules from frequent itemset
            self.CARS[k] = self.gen_rules(F[k], len_D)
            # Pruning rules
            self.CARS[k] = self.prune_rules(self.CARS[k], self.CARS[k - 1])

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
            if (class_count[major_class] / float(len_D)) >= self.min_sup:
                return {condset: (major_class, class_count)}

        return {}

    def gen_rules(self, F_k, len_D):

        rules = {}

        for condset, (major_class, class_count) in F_k.items():

            conf = class_count[major_class] / float(sum(class_count.values()))

            if conf > self.min_conf:
                # Checking if the support are the same for all classes
                if set(class_count.values()) == 1:
                    # Choose a random class
                    major_class = random.choice(class_count.keys())

                rules[condset] = (major_class, class_count)

        return rules

    def gen_itemset(self, F, df, target_col, itemsets):
        F_new = {}
        itemsets = set(itemsets)

        condset = list(F.keys())
        len_D = len(df)
        class_labels = df[target_col].unique()
        for i in range(len(condset)):
            cond = condset[i]
            temp = df.copy()
            cols = []
            for item in cond.items():
                value = item[1]
                col = item[0]
                temp = temp[temp[col] == value]
                cols.append(col)
            checked = []
            for j in range(i + 1, len(condset)):
                itemset = condset[j]
                # Apriori principle, where we merge 2 frequent superset into a candidate key
                # Checking if 2 itemsets differ only by 2 conditions
                itemset_keys = itemset.keys()
                cond_keys = cond.keys()
                if len(set(itemset_keys) - set(cond_keys)) == 1 and len(set(itemset.items()) ^ set(cond.items())) == 2:
                    variable = (set(itemset_keys) - set(cond_keys)).pop()
                    if variable not in checked:
                        checked.append(variable)
                        value = itemset.get(variable)

                        # Candidate generation
                        groupby = cols + [variable, target_col]

                        # Calculating frequency
                        group = temp.groupby(by=groupby).size().unstack(level=-1).reset_index()
                        group = group.fillna(0)
                        # Converting candidates in frequent itemset
                        group.apply(
                            lambda x: F_new.update(self.convert_to_rule_items(x, groupby[:-1], class_labels, len_D)),
                            axis=1)

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

                if subset_set < superset_set:
                    subset_value = r_[subset]

                    superset_class = superset_value[0]
                    superset_error = superset_value[1][superset_class] / sum(superset_value[1].values())

                    subset_class = subset_value[0]
                    subset_error = subset_value[1][subset_class] / sum(subset_value[1].values())

                    if superset_error >= subset_error:
                        to_remove.append(superset)
                        # break to as the superset has already been removed, no further testing is needed
                        break
        # print(len(r))
        for key in to_remove:
            r.pop(key)
        # print(len(r))
        return r
