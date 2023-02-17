#!/usr/bin/env python
# coding: utf-8

# Имеются результаты опроса студентов ТомГУ. Доступна информация:  
# 1. о физических параметрах опрашиваемых (возраст, рост, вес, пол, длина волос, размер обуви, год и месяц рождения, число удаленных зубов, наличие очков, разница в мм между средним и мизинцем/указательным/безымянными пальцами)  
# 2. о социальных параметрах (наличие сиблингов, количество друзей, расстояние до дома от универа, военнообязанность, номер этажа, размер родного города,  strange_people, social_network_duration_min, hostel) 
# 3. о предпочтениях (желаемое количество детей, любимые  животные, любимый фастфуд и шоколад, на что ставка в монетку и в игре "камень - ножницы - бумага")
# 4. об учебе (номер курса, факультет, оценки за ЕГЭ - при их наличии - по русскому, математике, физике, компьютерным наукам, химии, литературе, истории, географии, биологии, иностранному языку, социальным наукам, оценка в универе, наличие сложностей в учебе в последнем семестре, опоздания на первую пару в минутах, место в аудитории)  
# 5. остальные параметры (width_of_5000_mm, height_of_5000_mm)  
# 
# Попробуем решить следующие задачи с помощью метода решающих деревьев:  
# 1. Классификация объектов по полу,  
# 2. Задача регрессии (предсказание роста объекта)

# Начнем с задачи классификации объектов по полу в зависимости от их роста, веса, длины волос, ну и возьмем еще признак желаемое число детей

# In[1]:


# грузим либы
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import mean_absolute_error

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# считаем данные и выделим нужные для задачи признаки
df_train = pd.read_csv('df_train.csv')
df_train = df_train[['growth', 'weight', 'sex', 'hair_length', 'children_number']]


# In[7]:


# посмотрим на различия в распределении нецелевых признаков
sns.pairplot(df_train, hue = 'sex')


# In[11]:


# поищем оптимальные параметры для нашего дерева
param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'ccp_alpha': [0.1, .01, .001],
              'max_depth' : np.arange(1, 20),
              'criterion' :['gini', 'entropy']
             }
tree_clas = tree.DecisionTreeClassifier(random_state=0)
grid_search = GridSearchCV(estimator=tree_clas, param_grid=param_grid, cv=5, verbose=True)
grid_search.fit(df_train[['growth', 'weight', 'hair_length', 'children_number']], df_train.sex)


# In[12]:


# посмотрим на подобранные параметры
grid_search.best_estimator_


# In[14]:


# создаем модель, учим на трейне
model_tree = tree.DecisionTreeClassifier(ccp_alpha=0.001, max_depth=5, max_features='auto', random_state=0)
model_tree.fit(df_train[['growth', 'weight', 'hair_length', 'children_number']].values.reshape(-1,4), y = df_train.sex.values)


# In[15]:


# импортируем отрисовщик дерева
import graphviz
dot_data = tree.export_graphviz(model_tree, out_file = None,
                               feature_names = ['growth', 'weight', 'hair_length', 'children_number'],
                               class_names = ['f', 'm'],
                               filled = True, rounded = True,
                               special_characters = True)
graph = graphviz.Source(dot_data)
graph


# In[16]:


# грузим тест
df_test = pd.read_csv('df_test.csv')
# сформируем датасет для решения задачи предсказания пола по весу и росту
df_test_cut = df_test[['growth', 'weight', 'sex', 'hair_length', 'children_number']]
# удаляем пустые значения, они бесполезны
df_test_cut.dropna(inplace = True)
# гоним тест по предикту
df_test_cut['predicted'] = model_tree.predict(df_test_cut[['growth', 'weight', 'hair_length', 'children_number']].values.reshape(-1,4))
pd.crosstab(df_test_cut.predicted, df_test_cut.sex)


# In[38]:


# посмотрим на скоры
precision_recall_fscore_support(df_test_cut.predicted, df_test_cut.sex)


# In[21]:


# закодируем полученные результаты, пусть 0 - это верное предсказание,
# 1 - мужской пол определен как женский
# 2 - женский пол определен как мужской
df_test_cut['code'] = 0
df_test_cut.loc[(df_test_cut.sex == 'мужской')&(df_test_cut.predicted == 'женский'), 'code'] = '1'
df_test_cut.loc[(df_test_cut.sex == 'женский')&(df_test_cut.predicted == 'мужской'), 'code'] = '2'


# In[23]:


sns.scatterplot(data = df_test_cut, x = 'hair_length', y = 'growth', hue = 'code')


# In[24]:


sns.scatterplot(data = df_test_cut, x = 'weight', y = 'hair_length', hue = 'code')


# In[25]:


# кто эти люди, классифицированные ошибочно?
df_test_cut.query('code == "1"')


# In[36]:


sns.boxplot(data = df_test_cut.query('sex == "мужской"'), x = 'hair_length')


# In[35]:


sns.boxplot(data = df_test_cut.query('sex == "женский"'), x = 'hair_length')


# Алгоритм ошибся в двух мужчинах, имеющих длину волос более 15 см. Видимо, это максимально значимый признак. А что, если попробовать классифицировать объекты по длине волос и размеру обуви?

# In[39]:


# считаем данные и выделим нужные для задачи признаки
df_train = pd.read_csv('df_train.csv')
df_train = df_train[['sex', 'hair_length', 'shoe_size']]


# In[40]:


sns.pairplot(df_train)


# In[42]:


# поищем оптимальные параметры для нашего дерева
param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'ccp_alpha': [0.1, .01, .001],
              'max_depth' : np.arange(1, 20),
              'criterion' :['gini', 'entropy']
             }
tree_clas1 = tree.DecisionTreeClassifier(random_state=0)
grid_search = GridSearchCV(estimator=tree_clas1, param_grid=param_grid, cv=5, verbose=True)
grid_search.fit(df_train[['hair_length', 'shoe_size']], df_train.sex)


# In[43]:


# посмотрим на подобранные параметры
grid_search.best_estimator_


# In[44]:


# создаем модель, учим на трейне
model_tree1 = tree.DecisionTreeClassifier(ccp_alpha=0.01, max_depth=4, max_features='auto', random_state=0)
model_tree1.fit(df_train[['hair_length', 'shoe_size']].values.reshape(-1,2), y = df_train.sex.values)


# In[49]:


# грузим тест
df_test = pd.read_csv('df_test.csv')
# сформируем датасет для решения задачи предсказания пола по весу и росту
df_test_cut = df_test[['hair_length', 'shoe_size', 'sex']]
# удаляем пустые значения, они бесполезны
df_test_cut.dropna(inplace = True)
# гоним тест по предикту
df_test_cut['predicted'] = model_tree1.predict(df_test_cut[['hair_length', 'shoe_size']].values.reshape(-1,2))
pd.crosstab(df_test_cut.predicted, df_test_cut.sex)


# In[52]:


# оценим полученный результат
precision_recall_fscore_support(df_test_cut.predicted, df_test_cut.sex)


# In[ ]:


# закодируем полученные результаты, пусть 0 - это верное предсказание,
# 1 - мужской пол определен как женский
# 2 - женский пол определен как мужской
df_test_cut['code'] = 0
df_test_cut.loc[(df_test_cut.sex == 'мужской')&(df_test_cut.predicted == 'женский'), 'code'] = '1'
df_test_cut.loc[(df_test_cut.sex == 'женский')&(df_test_cut.predicted == 'мужской'), 'code'] = '2'


# In[54]:


# кто эти люди, классифицированные ошибочно?
df_test_cut.query('(code == "1") | (code == "2")')


# Те же самые мужчины с длинными волосами определены как женщины, плюс одна женщина с большим размером ноги классифицирована ошибочно

# ***Регрессия***   
# ***Задача предсказания роста объекта по его весу, длине волос и размеру обуви***

# In[11]:


# считаем данные и выделим нужные для задачи признаки
df_train = pd.read_csv('df_train.csv')
df_train = df_train[['growth', 'weight', 'hair_length', 'shoe_size']]


# In[12]:


sns.pairplot(df_train)


# In[13]:


# поищем оптимальные параметры для нашего дерева
param_grid = {"criterion": ["mse", "mae"],
              "max_depth": np.arange(1, 20),
              "min_samples_leaf": [20, 40, 100],
              "max_leaf_nodes": [5, 20, 100]
             }
tree_regr = tree.DecisionTreeRegressor(random_state=0)
grid_search = GridSearchCV(estimator=tree_regr, param_grid=param_grid, cv=5)
grid_search.fit(df_train[['weight', 'hair_length', 'shoe_size']], df_train.growth)


# In[14]:


# посмотрим на подобранные параметры
grid_search.best_estimator_


# In[15]:


# создаем дерево для задачи регрессии
model_tree_regr = tree.DecisionTreeRegressor(criterion='mse', max_depth=2, max_leaf_nodes=5, min_samples_leaf=20, random_state=0)
model_tree_regr.fit(df_train[['weight', 'hair_length', 'shoe_size']].values.reshape(-1, 3), y = df_train.growth.values)


# In[16]:


# импортируем отрисовщик дерева
import graphviz
dot_data = tree.export_graphviz(model_tree_regr, out_file = None,
                               feature_names = ['weight', 'hair_length', 'shoe_size'],
                               class_names = 'growth',
                               filled = True, rounded = True,
                               special_characters = True)
graph = graphviz.Source(dot_data)
graph


# In[22]:


# грузим тест
df_test = pd.read_csv('df_test.csv')
# сформируем датасет для решения задачи предсказания пола по весу и росту
df_test_cut = df_test[['growth', 'weight', 'hair_length', 'shoe_size']]
# гоним тест по предикту
df_test_cut['predicted'] = model_tree_regr.predict(df_test_cut[['weight', 'hair_length', 'shoe_size']].values.reshape(-1,3))


# In[23]:


# посмотрим на ошибку МАЕ
mean_absolute_error(df_test_cut.predicted, df_test_cut.growth)


# То есть, в среднем алгоритм ошибается в предсказании роста на 4,39 см.
