#!/usr/bin/env python
# coding: utf-8

# Имеются результаты опроса студентов ТомГУ. Доступна информация:  
# 1. о физических параметрах опрашиваемых (возраст, рост, вес, пол, длина волос, размер обуви, год и месяц рождения, число удаленных зубов, наличие очков, разница в мм между средним и мизинцем/указательным/безымянными пальцами)  
# 2. о социальных параметрах (наличие сиблингов, количество друзей, расстояние до дома от универа, военнообязанность, номер этажа, размер родного города,  strange_people, social_network_duration_min, hostel) 
# 3. о предпочтениях (желаемое количество детей, любимые  животные, любимый фастфуд и шоколад, на что ставка в монетку и в игре "камень - ножницы - бумага")
# 4. об учебе (номер курса, факультет, оценки за ЕГЭ - при их наличии - по русскому, математике, физике, компьютерным наукам, химии, литературе, истории, географии, биологии, иностранному языку, социальным наукам, оценка в универе, наличие сложностей в учебе в последнем семестре, опоздания на первую пару в минутах, место в аудитории)  
# 5. остальные параметры (width_of_5000_mm, height_of_5000_mm)  
# 
# Посмотрим, насколько хорошо можно решить **задачу классификации людей по росту и весу** методом kNN

# In[2]:


# грузим либы
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import tree
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[3]:


# считаем трейн
df_train = pd.read_csv('df_train.csv')


# In[17]:


# сформируем датасет для решения задачи предсказания пола по росту и весу
df_train = df[['growth', 'weight', 'sex']]


# In[18]:


df_train.head()


# In[105]:


# посмотрим на распределение опрашиваемых по весу, росту и полу
sns.scatterplot(data = df_train.query('weight > 20'), x = 'weight', y = 'growth', hue = 'sex')


# In[19]:


# поскольку выбранный метод предполагает необходимость нормировки:
# импортируем либу для предобработки данных
from sklearn.preprocessing import StandardScaler
# определяем модель из класса "Стандартный нормировщик"
scaler = StandardScaler()
# обучаем нормировщик, показываем ему данные
scaler.fit(df_train[['growth', 'weight']].values.reshape(-1, 2))
# нормируем нецелевые признаки
arr = scaler.transform(df_train[['growth', 'weight']].values.reshape(-1, 2))


# In[21]:


knn = KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV
k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)
  
# определим поисковик параметра k с наилучшими результатами
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False,verbose=1)
  
# обучим модели на k от 1 до 30
grid_search=grid.fit(df_train[['growth', 'weight']], df_train['sex'])


# In[22]:


# k для наилучшего результата
print(grid_search.best_params_)


# In[25]:


# создаем модель классификатора kNN с лучшим k = 14 и учим его на имеющейся выборке
weigth_klass_knn = KNeighborsClassifier(n_neighbors = 14)
weigth_klass_knn.fit(arr, y = df_train.sex.values)


# In[26]:


# загрузим тест
df_test = pd.read_csv('students_test.csv')
# исправим написание колонок, уберем пробел и заменим прописные буквы на строчные
df_test.columns = df_test.columns.str.replace(' ', '_')
df_test.columns = df_test.columns.str.lower()


# In[27]:


# сформируем датасет для решения задачи предсказания пола по весу и росту
df_test_cut = df_test[['growth', 'weight', 'sex']]


# In[90]:


# пропущены 21 значения веса опрашиваемых
df_test_cut.isna().sum()


# In[28]:


# поскольку по ним предсказаний сделать нельзя, удалим их из теста
df_test_cut.dropna(inplace = True)


# In[29]:


# аналогично трейну, нормируем данные
arr_test = scaler.transform(df_test_cut[['growth', 'weight']].values.reshape(-1,2))


# In[30]:


# прогоняем тест через предикт
df_test_cut['predicted'] = weigth_klass_knn.predict(arr_test)


# In[94]:


df_test_cut.head()


# **Оценим полученные результаты применения метода kNN** 

# In[31]:


# построим матрицу сопряженности
pd.crosstab(df_test_cut.predicted, df_test_cut.sex)


# In[33]:


print(accuracy_score(df_test_cut.predicted, df_test_cut.sex))


# In[34]:


print(precision_score(df_test_cut.predicted, df_test_cut.sex, average = None, zero_division = 1))


# In[35]:


print(recall_score(df_test_cut.predicted, df_test_cut.sex, average = None, zero_division = 1))


# In[36]:


# закодируем полученные результаты, пусть 0 - это верное предсказание,
# 1 - мужской пол определен как женский
# 2 - женский пол определен как мужской
df_test_cut['code'] = 0
df_test_cut.loc[(df_test_cut.sex == 'мужской')&(df_test_cut.predicted == 'женский'), 'code'] = '1'
df_test_cut.loc[(df_test_cut.sex == 'женский')&(df_test_cut.predicted == 'мужской'), 'code'] = '2'


# In[37]:


sns.scatterplot(data = df_test_cut, x = 'weight', y = 'growth', hue = 'code')


# Итак, двух мужчин алгоритм определил как женщин, оба они меньше 180 см ростом. Вобще, зона между 175 и 180 см в росте - самая ошибочная, тут находятся 2 мужчин, ошибочно определенные, и 5 женщин.
