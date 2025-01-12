#!/usr/bin/env python
# coding: utf-8

# Цель исследования - предсказать уровень удовлетворённости сотрудника на основе данных заказчика.
# 
# 
# План работ:
# 
# 1. Предобработка данных
# 1.1. Подгрузить данные
# 1.2. Посмотреть на пропуски в разных столбцах, и если таковых немного удалить
# 1.3. В оставшихся пропусках попробовать заполнить логически
# 1.4. Удалить явные дубликаты
# 1.5. Преобразовать типы данных
# 1.6. Обработать категориальные столбцы
# 1.7. Посмотреть на неявные дубликаты
# 
# 
# 2. Провести анализ данных
# 2.1. Построить графики признаков и посмотреть на распределения
# 2.2. Обработать выбросы
# 2.3. Возможно, какие-то признаки нужно будет категоризировать (если ненормальное распределение)
# 2.4. Рассмотреть на взаимосвзь между признаками друг с другом и между признаками и целевой переменной
# 
# 
# 3. Построить модель
# 3.1. сформировать пайплайн для моделей регрессии (спрогнозировать уровень удовлетворенности)
# 3.2. выбрать лучшую модель, проанализировать признаки (возможно, попытаться интерпретировать)
# 

# In[1]:


get_ipython().system('pip install phik  # -q убирает необязательные выводы в командах Linux')
get_ipython().system('pip install shap')
#!pip install -Uq matplotlib

get_ipython().system('pip install -U matplotlib')
get_ipython().system('pip install scikit-learn==1.4')


# In[2]:


get_ipython().system('pip install matplotlib==3.2.2')


# In[3]:


import pandas as pd
import numpy as np
#from math import sqrt
#import datetime as dt
import seaborn as sns

import phik
from phik import phik_matrix
from phik.report import plot_correlation_matrix

import scipy.stats as st
from scipy.stats import shapiro, ttest_ind

from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)
from matplotlib.pyplot import figure

#from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler#, RobustScaler, LabelEncoder

from sklearn.metrics import r2_score
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import recall_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer


# In[4]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
import shap


# # Задача 1

# ## Загрузка данных

# In[5]:


data_train = pd.read_csv('/datasets/train_job_satisfaction_rate.csv')
test_features = pd.read_csv('/datasets/test_features.csv')
test_target_job_satisfaction_rate = pd.read_csv('/datasets/test_target_job_satisfaction_rate.csv')


# In[6]:


data_train = data_train.set_index('id')
test_features = test_features.set_index('id')
test_target_job_satisfaction_rate = test_target_job_satisfaction_rate.set_index('id')


# In[7]:


data_test = test_features.merge(test_target_job_satisfaction_rate, on='id', how='left')
data_test.info()


# ##  Предобработка данных

# ###  train_job_satisfaction_rate

# In[8]:


data_train.head()


# In[9]:


data_train.info()


# <div class="alert alert-info"> типы столбцов сочетаются с одержанием, но есть пропуски (правда их количество не критично - можно просто удалить или заполлнить наиболее популярными значениями) <div>

# In[10]:


cols = ['dept', 'level', 'workload', 'last_year_promo', 'last_year_violations']


# In[11]:


for i in cols:
    print(data_train[i].unique())


# ###  test_features

# In[12]:


data_test.head()


# In[13]:


data_test.info()


# <div class="alert alert-info"> типы столбцов сочетаются с одержанием, но есть пропуски (правда их количество не критично - можно просто удалить или заполлнить наиболее популярными значениями) <div>

# In[14]:


for i in cols:
    print(data_test[i].unique())


# In[15]:


data_test.head()


# In[16]:


data_test.info()


# In[17]:


data_train.index.value_counts().sort_values(ascending = False)


# In[18]:


data_test.index.value_counts().sort_values(ascending = False)


# <div class="alert alert-info"> все id уникальны - соответственно удалять ничего нельзя (дубликатов нет) <div>

# По итогам предобработки данных:
# - колонки соовтествуют содержимому
# - пропусков есть - обработаем их в дальнейшем, заменив на моду
# - дубликатов нет
# - неявных дубликатов нет

# <div class="alert alert-info"> Сейчас все в пайплайне <div>

# ## Исследовательский анализ данных

# <div class="alert alert-info"> Рассмотрим количественные признаки <div>

# ###  data_train

# In[19]:


categorical_columns = [c for c in data_train.columns if data_train[c].dtype.name == 'object']
numerical_columns   = [c for c in data_train.columns if data_train[c].dtype.name != 'object']


# In[20]:


data_train.describe()


# <div class="alert alert-info"> Я тогда под каждым графиком сейчас чек лист буду оставлять по оформлению <div>

# <div class="alert alert-success">
# <h2> Комментарий ревьюера v.4 <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍</b> Учтено.

# In[21]:


data_train['employment_years'].hist(bins=10)
plt.title('Распределение Кол-во отраб лет')
plt.xlabel('Кол-во отраб лет')
plt.ylabel('Количество')
plt.rcParams["figure.figsize"] = (5,5)
plt.show()


# Названия осей Х и У - есть
# Название графика - есть

# In[22]:


data_train['supervisor_evaluation'].hist(bins=10)
plt.title('Распределение Оценка рук-ля')
plt.xlabel('Оценка рук-ля')
plt.ylabel('Количество')
plt.rcParams["figure.figsize"] = (5,5)
plt.show()


# Названия осей Х и У - есть
# Название графика - есть

# Убедились, что это категориальные признаки... посмотрим на них позже - в анализе категориальных

# In[23]:


print(data_train['employment_years'].unique())
print(data_train['supervisor_evaluation'].unique())


# In[24]:


data_train['salary'].hist(bins=10)
plt.title('Распределение Зарплата')
plt.xlabel('Зарплата')
plt.ylabel('Количество')
plt.rcParams["figure.figsize"] = (5,5)
plt.show()


# Названия осей Х и У - есть
# Название графика - есть

# In[25]:


data_train['job_satisfaction_rate'].hist(bins=10)
plt.title('Распределение Удовлетворенности')
plt.xlabel('Удовлетворенность')
plt.ylabel('Количество')
plt.rcParams["figure.figsize"] = (5,5)
plt.show()


# Названия осей Х и У - есть
# Название графика - есть

# In[26]:


numerical_columns = ['salary', 'job_satisfaction_rate']
categorical_columns = categorical_columns + ['employment_years', 'supervisor_evaluation']
for i in numerical_columns:
    plt.rcParams["figure.figsize"] = (5,5)
    plt.figure()
    plt.title("Распределение " + i)
    plt.ylabel('Уровень ' + i)
    plt.xlabel(i)
    plt.xticks([0], ['The Second Entry!'])
    plt.boxplot(data_train[i])


# Названия осей Х - есть
# Названия осей У - есть
# Название графика - есть

# In[27]:


res = pd.DataFrame(columns=['колонка', 'Результат'])
n=0
for i in numerical_columns:
    stat, p_val = shapiro(data_train[i])
    if p_val < 0.01: res.loc[n] = [i, 'не нормальное']
    else: res.loc[n] = [i, 'нормальное']
    n += 1
res


# <div class="alert alert-info"> Все признаки не имеют нормального распределения. Выбросов нет. Признак supervisor_evaluation и employment_years - носят категориальный характер <div>

# In[28]:


ax = data_train['employment_years'].value_counts().plot.bar()
plt.title('Распределение по Кол-во отраб лет')
plt.xlabel('Кол-во отраб лет')
plt.ylabel('Количество')
plt.ylabel('Количество')
plt.rcParams["figure.figsize"] = (5,5)
plt.show()


# Названия осей Х - есть
# Названия осей У - есть
# Название графика - есть

# In[29]:


ax = data_train['supervisor_evaluation'].value_counts().plot.bar()
plt.title('Распределение по Оценка рук-ля')
plt.xlabel('Кол-во Оценка рук-ля')
plt.ylabel('Количество')
plt.rcParams["figure.figsize"] = (5,5)
plt.show()


# Названия осей Х - есть
# Названия осей У - есть
# Название графика - есть

# <div class="alert alert-info"> Построим матрицу корреляций <div>

# In[30]:


phik_overview = data_train.phik_matrix(interval_cols=numerical_columns)
print(phik_overview.shape)


# In[31]:


plot_correlation_matrix(
    phik_overview.values,
    x_labels=phik_overview.columns,
    y_labels=phik_overview.index,
    vmin=0, vmax=1, color_map='Greens',
    title=r'correlation $\phi_K$',
    fontsize_factor=1.5,
    figsize=(20, 15)
)


# <div class="alert alert-info"> Мультиколлинеарности не наблюдается. Из инетресного - зарплата низко коррелирует с удовлетворенностью работой <div>

# In[32]:


phik_overview = data_test.phik_matrix(interval_cols=numerical_columns)
print(phik_overview.shape)


# In[33]:


plot_correlation_matrix(
    phik_overview.values,
    x_labels=phik_overview.columns,
    y_labels=phik_overview.index,
    vmin=0, vmax=1, color_map='Greens',
    title=r'correlation $\phi_K$',
    fontsize_factor=1.5,
    figsize=(20, 15)
)


# <div class="alert alert-info"> Мультиколлинеарности не наблюдается и на тестовой выборке. <div>

# ## Построение модели

# <div class="alert alert-info"> Для начала осуществим все преобразования, на основе выводов, по которым я пришел выше <div>
#     - категоризуем признак supervisor_evaluation и employment_years

# In[34]:


data_train.head()


# In[35]:


data_test.head()


# In[36]:


data_train.info()


# In[37]:


len(data_train)


# In[38]:


data_train = data_train.drop_duplicates()


# In[39]:


len(data_train)


# In[40]:


X_train = data_train.drop('job_satisfaction_rate', axis=1)
y_train = data_train['job_satisfaction_rate']
X_test = data_test.drop('job_satisfaction_rate', axis=1)
y_test = data_test['job_satisfaction_rate']


# In[41]:


X_train.info()


# In[42]:


ohe_columns = ['dept', 'last_year_promo', 'last_year_violations']
ord_columns = ['level', 'workload']#, 'supervisor_evaluation']


# In[43]:


num_columns = ['salary']


# In[44]:


just_columns = ['employment_years', 'supervisor_evaluation']


# <div class="alert alert-info"> Создадим pipeline <div>

# In[45]:


RANDOM_STATE = 42


# In[46]:


ohe_pipe=Pipeline([('simpleImputer_ohe', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
                 ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop = 'first'))])


# In[47]:


ord_pipe=Pipeline([('simpleImputer_before_ord', SimpleImputer(missing_values=np.nan, strategy = 'most_frequent')),
                 ('ord',  OrdinalEncoder(categories=[['junior', 'middle', 'sinior'],
                                                     ['low', 'medium', 'high'],],
                                                     #[1, 2, 3, 4, 5],], 
                handle_unknown='use_encoded_value', unknown_value=np.nan)),
                 ('simpleImputer_after_ord', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))])


# In[48]:


data_preprocessor=ColumnTransformer(transformers=[('ohe',ohe_pipe,ohe_columns),
                                    ('ord',ord_pipe,ord_columns),
                                    ('num',StandardScaler(),num_columns)],
                                    remainder='passthrough')


# In[49]:


pipe_final = Pipeline(
    [
        ('preprocessor', data_preprocessor),
        ('models', LinearRegression())
    ]
)


# In[50]:


X_test.info()


# In[51]:


for i in cols:
    print(X_test[i].unique())


# In[52]:


#X_test_p = pd.DataFrame(data_preprocessor.fit_transform(X_test))


# In[53]:


#feature_names = list(data_preprocessor.named_transformers_['ohe']['ohe'].get_feature_names_out(ohe_columns))
#feature_names += ord_columns
#feature_names += num_columns
#feature_names += just_columns
#X_test_p.columns = feature_names
#X_test_p.info()


# In[54]:


#X_test_p.head()


# In[55]:


y = pd.DataFrame(y_train)
y.info()


# In[56]:


param_grid = [
    {
        'models': [DecisionTreeRegressor(random_state=RANDOM_STATE)],
        'models__max_depth': range(2, 15),
        'models__max_features': range(2, 15),
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    },
    {
        'models': [LinearRegression()],
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']   
    }]


# <div class="alert alert-info"> Создадим функцию метрики <div>

# In[57]:


def smape( a , f ):
    smape = 1/ len (a) * np.sum (2 * np.abs (f-a) / (np.abs (a) + np.abs (f))*100)
    return smape


# In[58]:


scorer = make_scorer(smape, greater_is_better=False)  


# <div class="alert alert-info"> Поиск модели <div>

# In[59]:


grid = GridSearchCV(
    pipe_final, 
    param_grid=param_grid, 
    cv=5, 
    scoring=scorer, 
    n_jobs=-1)


# In[60]:


grid.fit(X_train, y_train)


# In[61]:


pd.set_option('display.max_colwidth', None)
result = pd.DataFrame(grid.cv_results_)
result['mean_test_score'] = result['mean_test_score'] * -1
display(result[['rank_test_score', 'param_models', 'mean_test_score','params']].sort_values('rank_test_score').head(10))


# In[62]:


pred_test = grid.best_estimator_.predict(X_test)
pred_test


# In[63]:


print(smape(pred_test, y_test))


# In[64]:


r2 = r2_score(y_test, pred_test)  
r2


# In[65]:


# проведите анализ остатков
# перед этим рассчитайте остатки
residuals = y_test - pred_test

# постройте графики по количественным признакам — гистограмму частотности 
# распределения и диаграмму рассеяния
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
axes[0].hist(residuals)
axes[0].set_title('Гистограмма распределения остатков')
axes[0].set_xlabel('Остатки')
axes[0].set_ylabel('КОличество')
#plt.rcParams["figure.figsize"] = (5,5)

axes[1].scatter(pred_test, residuals)
axes[1].set_xlabel('Предсказания модели')
axes[1].set_ylabel('Остатки')
axes[1].set_title('Анализ дисперсии')
#plt.rcParams["figure.figsize"] = (5,5)
plt.show()


# Названия осей Х - есть
# Названия осей У - есть
# Название графика - есть

# In[67]:


res = pd.DataFrame(columns=['колонка', 'Результат'])
stat, p_val = shapiro(residuals)
if p_val < 0.05: res.loc[n] = ['остатки', 'не нормальное']
else: res.loc[n] = ['остатки', 'нормальное']
res


# <div class="alert alert-info"> R2 = 0,86
# Распределение остатков:
# - смещения практически нет, то есть модель примерно адекватно оценивает
# - распределение остатков относительно равномерно

# <div class="alert alert-info"> Лучшая модель:
# {'models': DecisionTreeRegressor(random_state=42), 'models__max_depth': 13, 'models__max_features': 11, 'preprocessor__num': MinMaxScaler()}
# 
# SMAPE на тестовой выборке = 14,42
# R^2 = 0.86

# In[68]:


X_train.info()


# In[69]:


X_train_p = pd.DataFrame(data_preprocessor.fit_transform(X_train))
X = pd.DataFrame(X_train_p, columns= ohe_columns + ord_columns+num_columns)
feature_names = list(data_preprocessor.named_transformers_['ohe']['ohe'].get_feature_names_out(ohe_columns))
feature_names += ord_columns
feature_names += num_columns
feature_names += just_columns
X_train_p.columns = feature_names
X_train_p.info()


# In[70]:


model = grid.best_estimator_.named_steps['models']
#feature_importance = pd.DataFrame(model.feature_importances_)
feature_importance = pd.DataFrame({'Feature': X_train_p.columns, 'Importance': model.feature_importances_})
feature_importance = feature_importance.sort_values('Importance', ascending=True)
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
plt.title('Гистограмма важных признаков]')
plt.xlabel('Важность')
plt.ylabel('Наименование')
plt.rcParams["figure.figsize"] = (5,5)
plt.show()


# Названия осей Х - есть
# Названия осей У - есть
# Название графика - есть

# # Задача 2

# ## Загрузка данных

# In[71]:


train_quit = pd.read_csv('/datasets/train_quit.csv')
test_features = pd.read_csv('/datasets/test_features.csv')
test_target_quit = pd.read_csv('/datasets/test_target_quit.csv')


# In[72]:


train_quit.head()


# In[73]:


test_features.head()


# In[74]:


test_target_quit.head()


# In[75]:


test_target_quit.info()


# In[76]:


test_features.info()


# In[77]:


train_quit = train_quit.set_index('id')
test_features = test_features.set_index('id')
test_target_quit = test_target_quit.set_index('id')


# In[78]:


#Поскольку у нас тестовая выборка перекочевала из предыдущей задачи, то это наш X_test, в который нужно добавить признак quit
test_quit = X_test.merge(test_target_quit, how = 'left', on = 'id')
test_quit.info()


# ## Предобработка данных

# <div class="alert alert-info"> Предобрабатывать будем только train_quit, т.к. test_quit был исследован в предыдущей задаче

# In[79]:


train_quit.head()


# In[80]:


train_quit.info()


# типы столбцов сочетаются с одержанием, пропусков нет

# In[81]:


cols = ['dept', 'level', 'workload', 'last_year_promo', 'last_year_violations']
for i in cols:
    print(train_quit[i].unique())


# In[82]:


train_quit.index.value_counts().sort_values(ascending = False)


# По итогам предобработки данных:
# - колонки соовтествуют содержимому
# - пропусков нет
# - дубликатов нет
# - неявных дубликатов нет

# ## Исследовательский анализ данных

# In[83]:


train_quit.describe()


# In[84]:


train_quit['employment_years'].hist(bins=10)
plt.title('Распределение Кол-во отраб лет')
plt.xlabel('Кол-во отраб лет')
plt.ylabel('Количество')
plt.show()


# Названия осей Х - есть
# Названия осей У - есть
# Название графика - есть

# In[86]:


train_quit['supervisor_evaluation'].hist(bins=10)
plt.title('Распределение Оценка рук-ля')
plt.xlabel('Оценка рук-ля')
plt.ylabel('Количество')
plt.rcParams["figure.figsize"] = (5,5)
plt.show()


# По-прежнему, что это категориальные признаки... 

# Названия осей Х - есть
# Названия осей У - есть
# Название графика - есть

# In[87]:


print(train_quit['employment_years'].unique())
print(train_quit['supervisor_evaluation'].unique())


# In[88]:


train_quit['salary'].hist(bins=10)
plt.title('Распределение Зарплата')
plt.xlabel('Зарплата')
plt.ylabel('Количество')
plt.rcParams["figure.figsize"] = (5,5)
plt.show()


# Названия осей Х - есть
# Названия осей У - есть
# Название графика - есть

# In[89]:


plt.figure()
plt.title("Распределение зарплаты")
plt.rcParams["figure.figsize"] = (5,5)
plt.xlabel('salary')
plt.ylabel('Уровень зарплаты')
plt.xticks([0], ['The Second Entry!'])
#plt.boxplot('salary')
plt.boxplot(train_quit['salary']);


# Названия осей Х - есть
# Названия осей У - есть
# Название графика - есть

# Выбросов нет

# In[90]:


ax = train_quit['employment_years'].value_counts().plot.bar()
plt.title('Распределение по Кол-во отраб лет')
plt.xlabel('Кол-во отраб лет')
plt.ylabel('Количество')
plt.rcParams["figure.figsize"] = (5,5)
plt.show()


# Названия осей Х - есть
# Названия осей У - есть
# Название графика - есть

# In[91]:


ax = train_quit['supervisor_evaluation'].value_counts().plot.bar()
plt.title('Распределение по Оценка рук-ля')
plt.xlabel('Кол-во Оценка рук-ля')
plt.ylabel('Количество')
plt.rcParams["figure.figsize"] = (5,5)
plt.show()


# Названия осей Х - есть
# Названия осей У - есть
# Название графика - есть

# <div class="alert alert-info"> Построим матрицу корреляций <div>

# In[92]:


phik_overview = train_quit.phik_matrix(interval_cols=['salary'])
print(phik_overview.shape)


# In[93]:


plot_correlation_matrix(
    phik_overview.values,
    x_labels=phik_overview.columns,
    y_labels=phik_overview.index,
    vmin=0, vmax=1, color_map='Greens',
    title=r'correlation $\phi_K$',
    fontsize_factor=1.5,
    figsize=(20, 15)
)


# <div class="alert alert-info"> Мультиколлинеарности не наблюдается <div>

# In[94]:


len(train_quit)


# In[95]:


data_train = train_quit.drop_duplicates()


# In[96]:


len(train_quit)


# ## Портрет уволившегося

# <div class="alert alert-info"> Рассмотрим на портрет уволившегося и егго связь с остальными признаками <div>

# In[97]:


sns.pairplot(train_quit, hue = 'quit');


# In[98]:


table = train_quit[['dept','quit']].value_counts().reset_index()
table.columns = ['dept', 'quit', 'count']
table['share'] = table['count'].div(table.groupby('dept')['count'].transform(lambda x: x.sum()))
table.sort_values(by = ['dept','quit'])


# In[99]:


table = train_quit[['level','quit']].value_counts().reset_index()
table.columns = ['level', 'quit', 'count']
table['share'] = table['count'].div(table.groupby('level')['count'].transform(lambda x: x.sum()))
table.sort_values(by = ['level','quit'])


# In[100]:


table = train_quit[['workload','quit']].value_counts().reset_index()
table.columns = ['workload', 'quit', 'count']
table['share'] = table['count'].div(table.groupby('workload')['count'].transform(lambda x: x.sum()))
table.sort_values(by = ['workload','quit'])


# In[101]:


table = train_quit[['last_year_promo','quit']].value_counts().reset_index()
table.columns = ['last_year_promo', 'quit', 'count']
table['share'] = table['count'].div(table.groupby('last_year_promo')['count'].transform(lambda x: x.sum()))
table.sort_values(by = ['last_year_promo','quit'])


# In[102]:


table = train_quit[['last_year_violations','quit']].value_counts().reset_index()
table.columns = ['last_year_violations', 'quit', 'count']
table['share'] = table['count'].div(table.groupby('last_year_violations')['count'].transform(lambda x: x.sum()))
table.sort_values(by = ['last_year_violations','quit'])


# In[103]:


table = train_quit[['supervisor_evaluation','quit']].value_counts().reset_index()
table.columns = ['supervisor_evaluation', 'quit', 'count']
table['share'] = table['count'].div(table.groupby('supervisor_evaluation')['count'].transform(lambda x: x.sum()))
table.sort_values(by = ['supervisor_evaluation','quit'])


# In[104]:


table = train_quit[['employment_years','quit']].value_counts().reset_index()
table.columns = ['employment_years', 'quit', 'count']
table['share'] = table['count'].div(table.groupby('employment_years')['count'].transform(lambda x: x.sum()))
table.sort_values(by = ['employment_years','quit'])


# In[105]:


table = train_quit[['salary','quit']]
table['level_salary'] = ''
table.loc[table['salary']<=20000, ['level_salary']] = '0 - 20 000'
table.loc[(table['salary']>20000)&(table['salary']<=40000), ['level_salary']] = '20 000 - 40 000'
table.loc[(table['salary']>40000)&(table['salary']<=60000), ['level_salary']] = '40 000 - 60 000'
table.loc[(table['salary']>60000)&(table['salary']<=80000), ['level_salary']] = '60 000 - 80 000'
table.loc[table['salary']>80000, ['level_salary']] = '80 000 - 100 000'


# In[106]:


table = table[['level_salary','quit']].value_counts().reset_index()
table.columns = ['level_salary', 'quit', 'count']
table['share'] = table['count'].div(table.groupby('level_salary')['count'].transform(lambda x: x.sum()))
table.sort_values(by = ['level_salary','quit'])


# Можно сделать следующие выводы.
# Склонны к текучке:
# - сотрудники, которые проработали небольшой срок в компании - чем больше срок пребывания в компании, тем меньше вероятность уволиться
# - сотрудники, которые получают низкие оценки от руководства - в общем-то логично, вероятно, не справляются с задачами
# - низкооплачиваемые сотрудники и сортьрудники младших должностей. Эти две категории имеют устойчивую связь (чем выше должность - тем выше зарплата, логично) - с большей зарплатой сложнее найти достойный вариант, поэтому текучка там меньше
# - чем меньше загружен сотрудник работой, тем больше у него времени на исследование рынка труда и, вероятно, скучнее на работе, что уменьшает лояльность
# 
# Процент увольнения практически не зависит от депаратамента работы сотрудника
# 
# Сотрудники, не склонные к текучке:
# - получившие продвижение
# - не имеющие нарушений
# 

# In[107]:


test_quit.info()


# In[108]:


data_test[['job_satisfaction_rate']].info()


# In[109]:


hypo_table = test_quit.merge(data_test['job_satisfaction_rate'], how = 'left', on ='id')
hypo_table = hypo_table[['quit', 'job_satisfaction_rate']]
hypo_table.head()


# In[110]:


sns.pairplot(hypo_table, hue = 'quit');


# In[111]:


sns.kdeplot(data=hypo_table, x="job_satisfaction_rate", hue="quit");


# In[112]:


sns.histplot(data=hypo_table, stat='density', common_norm=False, hue="quit", x="job_satisfaction_rate")


# <div class="alert alert-info"> Судя по графику, чем больше удовлетворение от работы - тем меньше склонность увольняться
# <div class="alert alert-info"> Добавим этот признак к обучающим данным

# In[113]:


group1 = hypo_table[hypo_table['quit']=='yes']
group2 = hypo_table[hypo_table['quit']=='no']


ttest_ind(group1['job_satisfaction_rate'], group2['job_satisfaction_rate'])

#Ttest_indResult(statistic=-2.6034304605397938, pvalue=0.017969284594810425)


# <div class="alert alert-info"> 
# H 0 : µ 1 = µ 2 (две средние совокупности равны)
# H A : µ 1 ≠ µ 2 (две средние совокупности не равны)
# Поскольку p-значение нашего теста (pvalue=1.231122066517193e-104) меньше, чем альфа = 0,05, мы отвергаем нулевую гипотезу теста. 
# У нас достаточных данных, чтобы сказать, что средняя удовлетворенность между двумя популяциями различна.

# In[114]:


X_train.info()


# In[115]:


train_quit.info()


# In[116]:


train_quit_temp = train_quit.drop('quit', axis=1)
train_quit_temp.info()


# In[117]:


pred_satisfy = grid.best_estimator_.predict(train_quit_temp)
pred_satisfy


# In[118]:


train_quit['job_satisfaction_rate'] = pred_satisfy
train_quit.head()


# In[119]:


test_quit.info()


# In[120]:


train_quit.info()


# In[121]:


test_quit_temp = test_quit.drop('quit', axis=1)
test_quit_temp.info()


# In[122]:


pred_satisfy = grid.best_estimator_.predict(test_quit_temp)
pred_satisfy


# In[123]:


test_quit['job_satisfaction_rate'] = pred_satisfy
test_quit.head()


# ## Построение модели

# In[124]:


len(train_quit)


# In[125]:


train_columns = train_quit.drop('quit', axis=1)
train_columns = list(train_columns.columns)
train_columns


# In[126]:


train_quit = train_quit.drop_duplicates(train_columns)
len(train_quit)


# In[127]:


X_train = train_quit.drop('quit', axis=1)
y_train = train_quit['quit']
X_test = test_quit.drop('quit', axis=1)
y_test = test_quit['quit']


# In[128]:


len(X_train)


# In[129]:


X_train.head()


# In[130]:


X_test.head()


# In[131]:


ohe_columns = ['dept', 'last_year_promo', 'last_year_violations']
ord_columns = ['level', 'workload']#, 'supervisor_evaluation']
num_columns = ['salary', 'job_satisfaction_rate']#, 'employment_years']
just_columns = ['employment_years', 'supervisor_evaluation']


# In[132]:


RANDOM_STATE = 42


# In[133]:


ohe_pipe=Pipeline([('simpleImputer_ohe', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
                 ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop = 'first'))])


# In[134]:


ord_pipe=Pipeline([('simpleImputer_before_ord', SimpleImputer(missing_values=np.nan, strategy = 'most_frequent')),
                 ('ord',  OrdinalEncoder(categories=[['junior', 'middle', 'sinior'],
                                                     ['low', 'medium', 'high'],],
                                                     #[1, 2, 3, 4, 5],], 
                handle_unknown='use_encoded_value', unknown_value=np.nan)),
                 ('simpleImputer_after_ord', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))])


# In[135]:


data_preprocessor=ColumnTransformer(transformers=[('ohe',ohe_pipe,ohe_columns),
                                    ('ord',ord_pipe,ord_columns),
                                    ('num',StandardScaler(),num_columns)],
                                    remainder='passthrough')


# In[136]:


pipe_final = Pipeline(
    [
        ('preprocessor', data_preprocessor),
        ('models', DecisionTreeClassifier(random_state=RANDOM_STATE))
    ]
)


# In[137]:


param_grid = [
    {
        'models': [DecisionTreeClassifier(random_state=RANDOM_STATE)],
        'models__max_depth': range(1, 10),
        'models__max_features': range(1, 10),
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    },
    {
        'models': [KNeighborsClassifier()],
        'models__n_neighbors': range(1, 10),
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']   
    },
    {
        'models': [LogisticRegression(random_state=RANDOM_STATE,solver='liblinear',penalty='l1')],
        'models__C': range(1, 10),
        'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    }
]


# In[138]:


grid = GridSearchCV(
    pipe_final, 
    param_grid=param_grid, 
    cv=5, 
    scoring='roc_auc', 
    n_jobs=-1)


# In[139]:


grid.fit(X_train, y_train)


# In[140]:


pd.set_option('display.max_colwidth', None)
result = pd.DataFrame(grid.cv_results_)
display(result[['rank_test_score', 'param_models', 'mean_test_score','params']].sort_values('rank_test_score')) 


# In[141]:


probabilities = grid.best_estimator_.predict_proba(X_test)
probabilities_one = probabilities[:, 1]
print('Площадь ROC-кривой:', roc_auc_score(y_test, probabilities_one))


# In[142]:


strategies = ['most_frequent', 'stratified', 'uniform']#, 'constant'] 

test_scores = [] 
for s in strategies: 
    dclf = DummyClassifier(strategy = s, random_state = 0) 
    dclf.fit(X_train, y_train) 
    #score = dclf.score(X_test, y_test) 
    
    probabilities = dclf.predict_proba(X_test)
    probabilities_one = probabilities[:, 1]
    #print('Площадь ROC-кривой:', roc_auc_score(y_test, probabilities_one))
    
    score = roc_auc_score(y_test, probabilities_one)
    test_scores.append(score) 


# In[143]:


ax = sns.stripplot(strategies, test_scores); 
ax.set(xlabel ='Strategy', ylabel ='Test Score') 
plt.show();


# <div class="alert alert-info"> Результаты совсем грустные, и сильно отличаются от наших модельных. Делаю вывод, что модель адекватна

# In[144]:


X_train_p = pd.DataFrame(data_preprocessor.fit_transform(X_train))
#X = pd.DataFrame(X_train_p, columns= ohe_columns + ord_columns+num_columns)
feature_names = list(data_preprocessor.named_transformers_['ohe']['ohe'].get_feature_names_out(ohe_columns))
feature_names += ord_columns
feature_names += num_columns
feature_names += just_columns
X_train_p.columns = feature_names
X_train_p.info()


# In[145]:


X_train_p.columns


# In[146]:


model = grid.best_estimator_.named_steps['models']
#feature_importance = pd.DataFrame(model.feature_importances_)
feature_importance = pd.DataFrame({'Feature': X_train_p.columns, 'Importance': model.feature_importances_})
feature_importance = feature_importance.sort_values('Importance', ascending=True)
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
plt.title('Гистограмма важных признаков]')
plt.xlabel('Важность')
plt.ylabel('Наименование')
plt.rcParams["figure.figsize"] = (5,5)
plt.show()


# Названия осей Х - есть
# Названия осей У - есть
# Название графика - есть

# Ради интереса оставим только значимые признаки:
# - level
# - job_satisfaction_rate
# - workload
# - employment_years
# - supervisor_evaluation
# - salary

# ## Выводы

# Данные состоят из 2 выборок: обучающшей и тестовой.
# Каждая из выборок содержит следующие данные
# - данные c характеристикой сотрудников компании
# - данные с целевой переменной - удовлетворенностью от работы
# - данные с другой целевой переменной - увольнение сотрудника
# 
# На превом этапе была осуществлена предобработка данных:
# - были проверены пропуски в таблицах и таковые были обнаружены только в категориалных столбцах и заменены на наиболее популярное значение
# - не было обнаружено несочетающихся типов данных с содержанием столбцов
# - неявных и явных дубликатов обнаружено не было
# 
# На втором этапе был проведен статистический анализ данных:
# - были выявлены что ни один из непрерывных столбцов не носит нормальный характер
# - ряд непрерывных столбцов носит категориальный характер (оценка руководителя по шкале от 1 до 5)
# - выбросов обнаружено не было
# 
# По результатам предварительного анализа :
# - не было выявлено мультиколлинеарности
# - был построен Pipeline с моделями  KNeighborsClassifier, DecisionTreeClassifier, LogisticRegression
# 
# Для решения задачи по прогнозированию уровня удовлетворенности:
# - среди всех построенных моделей лучшей оказалась {'models': DecisionTreeRegressor(max_depth=14, max_features=13, random_state=42), 'models__max_depth': 14, 'models__max_features': 13, 'preprocessor__num': MinMaxScaler()}
# - Отбор моделей осущестлялся на основе метрики smape
# - Метрика лучшей модели на тренировочной выборке: 5.68
# - Метрика SMAPE на тестовой выборке: 14.32
# 
# 
# Для решения задачи по прогнозированию увольнения сотрудника:
# - в качестве одного из признаков был добавлен признак удовлетворенности, т.к. предварительный аналдиз показал зависимость целевого признака от этого фактора
# - среди всех построенных моделей лучшей оказалась {'models': DecisionTreeClassifier(max_depth=6, max_features=5, random_state=42), 'models__max_depth': 6, 'models__max_features': 5, 'preprocessor__num': 'passthrough'}
# - Отбор моделей осущестлялся на основе метрики roc-auc
# - Метрика лучшей модели на тренировочной выборке: 0.93
# - Площадь ROC-кривой на тестовой выборке: 0.92
# 
# Анализ наиболее значимых признаков выявил следующие факторы:
# Ради интереса оставим только значимые признаки:
# - level
# - job_satisfaction_rate
# - workload
# - employment_years
# - supervisor_evaluation
# - salary
# 
# 
# Можно сделать следующие выводы.
# Склонны к текучке:
# - сотрудники, которые проработали небольшой срок в компании - чем больше срок пребывания в компании, тем меньше вероятность уволиться
# - сотрудники, которые получают низкие оценки от руководства - в общем-то логично, вероятно, не справляются с задачами
# - низкооплачиваемые сотрудники и сортьрудники младших должностей. Эти две категории имеют устойчивую связь (чем выше должность - тем выше зарплата, логично) - с большей зарплатой сложнее найти достойный вариант, поэтому текучка там меньше
# - чем меньше загружен сотрудник работой, тем больше у него времени на исследование рынка труда и, вероятно, скучнее на работе, что уменьшает лояльность
# 
# Процент увольнения практически не зависит от депаратамента работы сотрудника
# 
# Сотрудники, не склонные к текучке:
# - получившие продвижение
# - не имеющие нарушений

# Из рекомендаций по снижению увольнений я бы посоветовал следующее:
# Двумя основными факторами, оказывающими влияние на принятие решения по увольнению являются удовлетворенность от работы и количество проработанных в компании лет.
# 
# Касательно количества лет - то, наиболее высокие риски у новых сотрудников - работающих в компании менее одного года. Вероятно, это адлаптационный период, который определяет сознание сотрудника  - оставаться в компании или нет. Соответственно компании нужно "пристально" следить за новыми сотрудниками.
# Это можно сделать, например, введя институт наставничества, чтобы обеспечить наиболее мягкое погружение новых сотрудников в среду компании.
# 
# Касательно удовлетворенности от работы - то наиболее важными факторами в большей степени, определяющими уровень удовлетворенности является оценка руководителя, в меньшей степени  - уровень зарплаты. Влияние остальных признаков еще меньше.
# Поэтому институт наставничества становится логичен и в рамках оценки руководителя. Руководителям необходимо быть более лояльными (особенно в отношении "молодых" сотрудников - со стажем до года), и также в большей степени применять позитивное стимулирование в целом своих сотрудников.
# Для того, чтобы снизить текучку - необходимо также пересмотреть уровень оплаты труда. Как показала статистика, сотрудники с окладом более 40 тыс. менее склонны к увольнениям.
