import pandas as pd
import sklearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import joblib

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', None)


def preprocess_data(data, imputer=None, scaler=None):
    data['Status'] = data['Status'].apply(lambda x: 0 if x == 'Developing' else 1)
    data['Country'] = pd.factorize(data['Country'])[0].astype(np.uint16)
    return fill_nan_data_by_line_reg_all(data, data.corr())


def single_linear_regression(data, x_name, y_name):
    #print(data)
    data_x = data[x_name].values.reshape(-1, 1)
    data_y = data[y_name].values
    reg = LinearRegression()
    reg.fit(data_x, data_y)
    #sns.scatterplot(data=data, x=x_name, y=y_name)
    plt.show()
    return reg


def multi_linear_regression(data_x, data_y):
    #print(data.shape)
    #data_x = data[data.columns[0 : data.shape[1] - 1]]
    #data_y = data.iloc[:, -1]
    reg = LinearRegression()
    reg.fit(data_x, data_y)
    #print(reg.intercept_)
    #print(reg.coef_)
    return reg


def model_loss(data_x, data_y, reg):
    data_processed = preprocess_data(data_x)
    #data_x = data_processed[data_processed.columns[0: data_processed.shape[1] - 1]]
    #data_y = data_processed.iloc[:, -1]
    predict_y = reg.predict(data_processed)

    mes = mean_squared_error(data_y, predict_y)
    r2 = r2_score(data_y, predict_y)
    print('mes=%f' % mes)
    print('R2=%f' % r2)


def single_line_reg_predict(regression, predict_data):
    return regression.predict(predict_data)


def multi_line_reg_predict(regression, predict_data):
    return regression.predict(predict_data)


def max_corr_by_name(data, corr, name):
    corr_name = pd.DataFrame(corr, columns=[name])
    corr_name_abs = corr_name.iloc[corr_name[name].abs().argsort()[::-1]]
    for idx, row in corr_name_abs.iterrows():
        if idx != name and not data[idx].isnull().any():
            return idx
    return None


def fill_nan_data_by_line_reg(train_data, grouped_by_data, corr, name):
    max_corr_name = max_corr_by_name(grouped_by_data, corr, name)
    data_by_name = grouped_by_data[[max_corr_name, name]]
    data_dropped = data_by_name.dropna()
    if data_dropped.shape[0] < 2 or data_dropped.shape[0] == data_by_name.shape[0]:
        return

    data_isnull = data_by_name[data_by_name.isnull().T.any()]
    reg = single_linear_regression(data_dropped, max_corr_name, name)
    predict_values = single_line_reg_predict(reg, data_isnull[max_corr_name].values.reshape(-1, 1))

    i = 0
    for idx, row in data_isnull.iterrows():
        train_data.loc[idx, name] = predict_values[i]
        ++i


def fill_nan_data_by_line_reg_all(train_data, corr):
    nan_names = []
    for kv in train_data.isnull().sum().items():
        if kv[1] > 0:
            nan_names.append(kv[0])

    train_data_grouped_by_country = train_data.groupby('Country')

    for country, group in train_data_grouped_by_country:
        for nan_name in nan_names:
            fill_nan_data_by_line_reg(train_data, group, corr, nan_name)
        after_data = train_data.groupby('Country').get_group(country)
        #if np.any(after_data.isnull()):
            #print(after_data)
            #print()

    train_data_dropped = train_data.drop(labels=['Hepatitis B', 'GDP', 'Population', ' thinness 5-9 years'], axis=1)
    nan_names_dropped = []
    for kv in train_data_dropped.isnull().sum().items():
        if kv[1] > 0:
            nan_names_dropped.append(kv[0])
    for name in nan_names_dropped:
        train_data_dropped[name] = train_data_dropped[name].fillna(train_data_dropped.groupby('Country')[name].transform('mean'))

    train_data_dropped.fillna(train_data_dropped.mean(), inplace=True)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(train_data_dropped)
    train_data_scaler = pd.DataFrame(scaler.transform(train_data_dropped), columns=train_data_dropped.columns)
    return train_data_scaler


if __name__ == '__main__':
    data = pd.read_csv('./data/train_data.csv')
    data_x = data.drop(['Adult Mortality'], axis=1)
    data_y = data['Adult Mortality']
    train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(data_x, data_y, train_size=0.8, random_state=30)
    processed_data = preprocess_data(train_data_x)
    reg = multi_linear_regression(processed_data, train_data_y)
    print(reg.intercept_)
    print(reg.coef_)
    model_loss(train_data_x, train_data_y, reg)

    processed_test_data_x = preprocess_data(test_data_x)
    predict_y = reg.predict(processed_test_data_x)
    vs = pd.DataFrame()
    vs['real'] = test_data_y
    vs['predict'] = predict_y
    print(vs)

