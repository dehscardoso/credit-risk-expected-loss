import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as stat
from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score


# função para converter data string em meses
def converte_colunas_data(df, coluna):
    data_ref = pd.to_datetime('2020-12-31')
    df[coluna] = pd.to_datetime(df[coluna], format='%b-%Y') 
    df['mths_since_' + coluna] = round(pd.to_numeric((data_ref - df[coluna]) / np.timedelta64(1, 'M')))
    df['mths_since_' + coluna] = df['mths_since_' + coluna].apply(lambda x: df['mths_since_' + coluna].max() if x < 0 else x)
    df.drop(columns = [coluna], inplace = True)


# função para cálculo tabular do peso de evidência e valor da informação
def iv_woe(data, target, bins=10, show_woe=False):
    newDF, woeDF = pd.DataFrame(), pd.DataFrame()
    cols = data.columns   
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Eventos']
        d['%_Eventos'] = np.maximum(d['Eventos'], 0.5) / d['Eventos'].sum()
        d['Nao_Eventos'] = d['N'] - d['Eventos']
        d['%_Nao_Eventos'] = np.maximum(d['Nao_Eventos'], 0.5) / d['Nao_Eventos'].sum()
        d['WoE'] = np.log(d['%_Eventos']/d['%_Nao_Eventos'])
        d['IV'] = d['WoE'] * (d['%_Eventos'] - d['%_Nao_Eventos'])
        d.insert(loc=0, column='Variavel', value=ivars)
        temp =pd.DataFrame({"Variavel" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variavel", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)
        if show_woe == True:
            print(d)
    return newDF, woeDF
  

# função para distribuição e plotagem de feature 
def dist(df, feature):
    plt.figure(figsize=(4,2))
    sns.violinplot(df[feature], color='lightblue')
    print('Número de valores únicos:', df[feature].nunique())
    print('Distribuição:')
    print(df[feature].describe().T)


# função para cálculo tabular do peso de evidência e valor da informação para variáveis categóricas 
def woe_categorica(df, variaveis_discretas, variavel_target):
    df = pd.concat([df[variaveis_discretas], variavel_target], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_bons']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_bons'] = df['prop_bons'] * df['n_obs']
    df['n_ruins'] = (1 - df['prop_bons']) * df['n_obs']
    df['prop_n_bons'] = df['n_bons'] / df['n_bons'].sum()
    df['prop_n_ruins'] = df['n_ruins'] / df['n_ruins'].sum()
    df['WoE'] = np.log(df['prop_n_bons'] / df['prop_n_ruins'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop = True)
    df['diff_prop_bons'] = df['prop_bons'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_bons'] - df['prop_n_ruins']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df


# função para cálculo tabular do peso de evidência e valor da informação para variáveis contínuas
def woe_continua(df, variaveis_continuas, variavel_target):
    df = pd.concat([df[variaveis_continuas], variavel_target], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_bons']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_bons'] = df['prop_bons'] * df['n_obs']
    df['n_ruins'] = (1 - df['prop_bons']) * df['n_obs']
    df['prop_n_bons'] = df['n_bons'] / df['n_bons'].sum()
    df['prop_n_ruins'] = df['n_ruins'] / df['n_ruins'].sum()
    df['WoE'] = np.log(df['prop_n_bons'] / df['prop_n_ruins'])
    df['diff_prop_bons'] = df['prop_bons'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_bons'] - df['prop_n_ruins']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df

# função para plotar peso de evidência
def plot_woe(df, rotacao_x_axis = 0):
    x = np.array(df.iloc[:, 0].apply(str))
    y = df['WoE']
    plt.figure(figsize=(20, 5))
    plt.plot(x, y, marker = 'o', linestyle = '--', color = 'k')
    plt.xlabel(df.columns[0], fontsize=13)
    plt.ylabel('Peso de Evidência', fontsize=13)
    plt.title(str('Peso de Evidência: ' + df.columns[0]), fontsize=18, pad=25)
    plt.xticks(rotation = rotacao_x_axis)


# classe para cálculo da regressão logística
class LogisticRegression_with_p_values:
    
    def __init__(self,*args,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)

    def fit(self,X,y):
        self.model.fit(X,y)
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X) 
        Cramer_Rao = np.linalg.inv(F_ij) 
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates 
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores] 
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values


# classe para cálculo da regressão logística
class LinearRegression(linear_model.LinearRegression):
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=1, positive=False):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive
        
    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)
        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
        self.t = self.coef_ / se
        self.p = np.squeeze(2 * (1 - stat.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1])))
        return self