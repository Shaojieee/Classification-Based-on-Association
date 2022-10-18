from CBA_RG import *
from CBA_CB import *
import pandas as pd
import time

def main():
    csv_files = {
        './data/bank_churners/clean_bank_churners.csv': 'Attrition_Flag',
        './data/breast_w/clean_breast_w.csv': 'class',
        './data/customer_segmentation/clean_customer_segmentation.csv': 'Segmentation',
        './data/german/clean_german.csv': 'good/bad customer (response)',
        # './data/signs/clean_signs.csv': 'word',
        # './data/vehicle/clean_vehicle.csv': 'Type of Vehicle',
        # './data/wine/clean_wine.csv': 'Class',
        './data/zoo/clean_zoo.csv': 'type'
    }

    for csv, target_col in csv_files.items():
        data = pd.read_csv(csv)
        start = time.time()
        rule_gen = RuleGenerator(min_sup=0.01, min_conf=0.5)
        rule_gen.generate_rules(data, target_col)

        classifier = Classifier(rule_gen)
        classifier.build_classifier(data, target_col)

        ans = classifier.predict(data)
        end = time.time()
        if ans is not None:
            ans['correct'] = (ans[target_col] == ans['prediction'])
            print('==============================')
            print(f'For {csv}')
            print(ans['correct'].value_counts())
            print(f'Accuracy: {len(ans[ans["correct"]==True])/len(ans)}')
            print(f'Total Time: {end-start}')
            print('==============================')


if __name__=='__main__':
    main()
