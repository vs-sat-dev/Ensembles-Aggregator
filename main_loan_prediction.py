import pandas as pd
import numpy as np

from eaggregator import EnsemblesAggregator


def factorize_cat_features(data, factorize_map=None):
    for column in data.columns:
        if data[column].dtype == 'object':
            if factorize_map:
                data.loc[:, column] = data[column].replace(factorize_map[column])
            else:
                data.loc[:, column] = data.loc[:, column].factorize()[0]
    return data


if __name__ == '__main__':
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    submission = df_test[['Loan_ID']]

    df_train = df_train.drop('Loan_ID', axis=1)
    df_train.loc[:, 'Dependents'] = df_train.loc[:, 'Dependents'].replace('3+', '3').fillna(0).astype('int')

    df_test = df_test.drop('Loan_ID', axis=1)
    df_test.loc[:, 'Dependents'] = df_test.loc[:, 'Dependents'].replace('3+', '3').fillna(0).astype('int')

    target_feature = 'Loan_Status'

    if target_feature:
        df_train[target_feature] = df_train[target_feature].replace({'N': 0, 'Y': 1})

    df_train = factorize_cat_features(df_train)

    df_test = factorize_cat_features(df_test)

    x = df_train.drop(target_feature, axis=1)
    y = df_train[[target_feature]]

    print(f'xlen: {len(x)}')
    print(f'ylen: {len(y)}')

    ea = EnsemblesAggregator(x, y, objective_type='binary')
    ea.fit(num_trials=100, models_types=['xgboost'])
    preds = ea.predict(df_test)

    preds = np.rint(preds)
    submission['Loan_Status'] = preds.tolist()
    submission['Loan_Status'] = submission['Loan_Status'].replace({0: 'N', 1: 'Y'})
    submission.to_csv('submission.csv', index=False)



