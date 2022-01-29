import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import numpy as np

from eaggregator import EnsemblesAggregator


def prepare_data(data):
    data['date'] = pd.to_datetime(data['date'])
    data['day'] = data['date'].dt.day
    data['day_of_week'] = data['date'].dt.day_of_week
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    data['dayofyear'] = data['date'].dt.dayofyear

    cal = calendar()
    holidays = cal.holidays(start=data['date'].min(), end=data['date'].max())
    data['holiday'] = data['date'].isin(holidays)

    data['country'] = data['country'].factorize()[0]
    data['store'] = data['store'].factorize()[0]
    data['product'] = data['product'].factorize()[0]

    data = data.drop(['row_id', 'date'], axis=1)

    return data


def smape(target, predicted):
    return 100/len(target) * np.sum(2 * np.abs(predicted - target) / (np.abs(target) + np.abs(predicted)))


if __name__ == '__main__':
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    submission = df_test[['row_id']]

    df_train = prepare_data(df_train)
    df_test = prepare_data(df_test)

    target_feature = 'num_sold'

    x = df_train.drop(target_feature, axis=1)
    y = df_train[[target_feature]]

    ea = EnsemblesAggregator(x, y, objective_type='regression', evaluation_func=smape)
    ea.fit(num_trials=1, models_types=['tabnet'])
    preds = ea.predict(df_test)

    submission[target_feature] = preds.tolist()
    submission.to_csv('submission.csv', index=False)



