import numpy as np
from CBA_CB import *
from CBA_RG import *


class Adaboost:

    def __init__(self, T=50):
        self.T = T
        self.models = []
        self.stumps = {}

    # Train T number of CBA_CB that only has rules of 1 item (Weak Trees)
    # After training 1 tree, the weights of each row in the dataset will be rebalanced so that the next tree will be more likely to train on samples that are getting wrongly predicted
    def train(self, train_df, target_col, min_sup=0.01, min_conf=0.5, prune_rules=True):
        weights = np.ones(len(train_df)) / len(train_df)
        train_variables = list(train_df.columns)
        train_variables.remove(target_col)
        for t in range(self.T):
            self.stumps[t] = {}
            if t != 0:
                temp_df = train_df.sample(frac=1, replace=True, weights=weights, random_state=42)
            else:
                temp_df = train_df.copy()
            for variable in train_variables:
                stump_df = temp_df[[variable, target_col]]
                rule_gen = RuleGenerator(min_sup=min_sup, min_conf=min_conf)
                total_CARs += sum([len(x) for x in rule_gen.CARS.values()])
                classifier = Classifier(rule_gen)
                classifier.build_classifier(stump_df, target_col)

                # TODO: Ugly Code! Need to fix the whole no rules thing
                # For case if no rules generated
                if len(classifier.rules) == 0:
                    continue
                ans = classifier.predict(train_df[[variable, target_col]])
                ans['correct'] = (ans[target_col] == ans['prediction'])
                acc = len(ans[ans['correct'] == True]) / len(ans)

                self.stumps[t][variable] = {'acc': acc, 'model': classifier}

            max_acc_variable = max(self.stumps[t].items(), key=lambda x: x[1]['acc'])
            #     print(max_acc_variable)

            ans = max_acc_variable[1]['model'].predict(train_df[[max_acc_variable[0], target_col]])
            ans['wrong'] = (ans[target_col] != ans['prediction'])

            error = np.dot(weights, (ans['wrong'])) / sum(weights)
            alpha = np.log((1 - error) / error)

            weights = weights * np.exp(alpha * ans['wrong'])
            weights = weights / sum(weights)
            self.models.append({'alpha': alpha, 'model': max_acc_variable[1]['model'], 'variable': max_acc_variable[0]})

        print(f'Total Number of rules in adaboost {self.T} trees: {sum([len(x["model"].rules) for x in self.models])}')
        
        return self

    def predict(self, test_df):
        ans = pd.DataFrame(index=test_df.index)
        for i in range(len(self.models)):
            model = self.models[i]

            temp = model['model'].predict(test_df)
            classes = set(temp['prediction'].unique()) - set(ans.columns)
            if classes:
                ans = ans.join(pd.DataFrame(columns=list(classes)))
                ans = ans.fillna(0)
            ans = ans + pd.get_dummies(temp['prediction']) * model['alpha']
        ans['prediction'] = ans.idxmax(axis='columns')
        ans = test_df.merge(ans, left_index=True, right_index=True)
        return ans
