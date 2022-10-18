import pandas as pd
import numpy as np
from frozendict import frozendict
import random


class Classifier:

    def __init__(self, rule_builder):
        self.rule_builder = rule_builder
        self.rules = None
        self.sorted_CARS = None

    def sort_rules(self, len_D):
        sorted_CARS = []
        for values in self.rule_builder.CARS.values():
            sorted_CARS.extend(list(values.items()))
        self.sorted_CARS = sorted(sorted_CARS, key=lambda x: (x[1][1][x[1][0]] / sum(x[1][1].values()), x[1][1][x[1][0]] / len_D, len(x[0])), reverse=True)

    def build_classifier(self, df, target_col):
        len_D = len(df)
        self.sort_rules(len_D)
        temp_df = df
        rules = []

        if len(self.sorted_CARS)==0:
            # print('No CARS from rule generator!')
            self.rules = []
            return
        for CARS in self.sorted_CARS:
            cond, result = CARS
            cond_df = temp_df.loc[(temp_df[list(cond)] == pd.Series(cond)).all(axis=1)]
            correct = cond_df[cond_df[target_col] == result[0]]

            if len(correct) != 0:
                temp_df = temp_df.drop(index=cond_df.index)
                if len(temp_df) == 0:
                    default_class = random.choice(df[target_col].unique())
                else:
                    default_class = temp_df[target_col].value_counts().idxmax()
                total_error = (len(cond_df) - len(correct)) + len(temp_df[temp_df[target_col] != default_class])
                error = {'default': len(temp_df[temp_df[target_col] != default_class]),
                         'class': (len(cond_df) - len(correct))}
                rules.append([CARS, default_class, total_error, error])

        lowest_error_id = np.argmin([x[2] for x in rules])
        pruned_rules = rules[:lowest_error_id + 1]
        self.rules = pruned_rules

    def predict(self, df):
        temp_df = df.copy()
        ans = df.copy()
        # Setting all to default class
        if len(self.rules)==0:
            # print('No rules!')
            return None
        ans['prediction'] = self.rules[-1][1]
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
