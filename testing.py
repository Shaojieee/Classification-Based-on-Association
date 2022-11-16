from CBA_RG import *
from CBA_CB import *
from Adaboost import *
import pandas as pd
import time
from sklearn.model_selection import train_test_split


def main():
    csv_files = {
        # UCI Dataset
        './data/breast_w/clean_breast_w.csv': 'class',
        './data/german/clean_german.csv': 'class',
        './data/zoo/clean_zoo.csv': 'type',
        './data/tic_tac_toe/clean_tic-tac-toe.csv': 'class',
        './data/hepatitis/clean_hepatitis.csv': 'class',
        # Kaggle Dataset
        './data/mobile/clean_mobile.csv': 'price_range',
        './data/stroke/clean_stroke.csv': 'stroke',
        './data/gender/gender.csv': 'Gender'
    }

    for csv, target_col in csv_files.items():
        data = pd.read_csv(csv, index_col=0)
        train_df, test_df = train_test_split(data, test_size=0.1, random_state=42)
        start = time.time()
        rule_gen = RuleGenerator(min_sup=0.01, min_conf=0.5)
        rule_gen.generate_rules(train_df, target_col, prune_rules=True)
        print(f'Total number of CARS in CBA-RG: {sum([len(x) for x in rule_gen.CARS.values()])}')
        classifier = Classifier(rule_gen)
        classifier.build_classifier(train_df, target_col)
        end = time.time()
        train_ans = classifier.predict(train_df)
        test_ans = classifier.predict(test_df)

        if train_ans is not None:
            train_ans['correct'] = (train_ans[target_col] == train_ans['prediction'])
            test_ans['correct'] = (test_ans[target_col] == test_ans['prediction'])
            print('==============================')
            print(f'For {csv}')
            print('Single Classifier')
            print('Train')
            print(train_ans['correct'].value_counts())
            print(f'Accuracy: {len(train_ans[train_ans["correct"]==True])/len(train_ans)}')
            print('Test')
            print(test_ans['correct'].value_counts())
            print(f'Accuracy: {len(test_ans[test_ans["correct"] == True]) / len(test_ans)}')
            print(f'Total Fit Time: {end-start}')
            print('==============================')

        adaboost = Adaboost(T=70)
        start = time.time()
        adaboost.train(train_df, target_col, prune_rules=True)
        end = time.time()
        adaboost_train_ans = adaboost.predict(train_df)
        adaboost_test_ans = adaboost.predict(test_df)
        if adaboost_train_ans is not None:
            adaboost_train_ans['correct'] = (adaboost_train_ans[target_col] == adaboost_train_ans['prediction'])
            adaboost_test_ans['correct'] = (adaboost_test_ans[target_col] == adaboost_test_ans['prediction'])
            print('==============================')
            print(f'For {csv}')
            print('ADABOOST')
            print('Train')
            print(adaboost_train_ans['correct'].value_counts())
            print(f'Accuracy: {len(adaboost_train_ans[adaboost_train_ans["correct"]==True])/len(adaboost_train_ans)}')
            print('Test')
            print(adaboost_test_ans['correct'].value_counts())
            print(f'Accuracy: {len(adaboost_test_ans[adaboost_test_ans["correct"] == True]) / len(adaboost_test_ans)}')
            print(f'Total Fit Time: {end-start}')
            print('==============================')


if __name__=='__main__':
    main()
