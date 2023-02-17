#!/usr/bin/env python
# coding: utf-8

# Имеются результаты опроса студентов ТомГУ. Доступна информация:  
# 1. о физических параметрах опрашиваемых (возраст, рост, вес, пол, длина волос, размер обуви, год и месяц рождения, число удаленных зубов, наличие очков, разница в мм между средним и мизинцем/указательным/безымянными пальцами)  
# 2. о социальных параметрах (наличие сиблингов, количество друзей, расстояние до дома от универа, военнообязанность, номер этажа, размер родного города,  strange_people, social_network_duration_min, hostel) 
# 3. о предпочтениях (желаемое количество детей, любимые  животные, любимый фастфуд и шоколад, на что ставка в монетку и в игре "камень - ножницы - бумага")
# 4. об учебе (номер курса, факультет, оценки за ЕГЭ - при их наличии - по русскому, математике, физике, компьютерным наукам, химии, литературе, истории, географии, биологии, иностранному языку, социальным наукам, оценка в универе, наличие сложностей в учебе в последнем семестре, опоздания на первую пару в минутах, место в аудитории)  
# 5. остальные параметры (width_of_5000_mm, height_of_5000_mm)  
# 
# Посмотрим, насколько хорошо можно решить **задачу классификации людей по росту и весу** методом линейного классификатора

# In[15]:


# грузим либы
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import tree
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit


# In[2]:


# считаем трейн
df_train = pd.read_csv('df_train.csv')


# In[4]:


# сформируем датасет для решения задачи предсказания пола по росту и весу
df_train = df_train[['growth', 'weight', 'sex']]


# In[5]:


df_train.head()


# In[6]:


# посмотрим на распределение опрашиваемых по весу, росту и полу
sns.scatterplot(data = df_train, x = 'weight', y = 'growth', hue = 'sex')


# Линейный классификатор подразумевает, что классы можно разделить между собой прямой линией (необходимо построить максимально разделяющую прямую). Ему также требуется *нормировка*

# In[7]:


# поскольку метод предполагаtт необходимость нормировки:
# импортируем либу для предобработки данных
from sklearn.preprocessing import StandardScaler
# определяем модель из класса "Стандартный нормировщик"
scaler = StandardScaler()
# обучаем нормировщик, показываем ему данные
scaler.fit(df_train[['growth', 'weight']].values.reshape(-1, 2))
# нормируем нецелевые признаки
arr = scaler.transform(df_train[['growth', 'weight']].values.reshape(-1, 2))


# In[9]:


# создаем модель классификатора
weigth_klass = SGDClassifier(random_state = 0, tol=1e-3)


# In[10]:


# Создание сетки гиперпараметров
# будем подбирать коэффициент регуляризации, шаг градиентного спуска, количество итераций и параметр скорости обучения.
parameters_grid = {
       'alpha' : np.linspace(0.00001, 0.0001, 15),
       'learning_rate': ['optimal', 'constant', 'invscaling'],
       'eta0' : np.linspace(0.00001, 0.0001, 15),
       'max_iter' : np.arange(5,10),
   }


# In[19]:


# Создание экземпляра класса кросс-валидации
cv = StratifiedShuffleSplit(n_splits=10, test_size = 0.2)
# Создание экземпляра GridSearch (из sklearn)
# Первый параметр — модель, второй — сетка гиперпараметров, третий — функционал ошибки, четвертый — кросс-валидация
grid_cv = GridSearchCV(weigth_klass, parameters_grid, scoring = 'accuracy', cv = cv)
grid_cv.fit(arr, df_train.sex)


# In[20]:


# лучшая модель
print(grid_cv.best_estimator_)


# In[21]:


# ошибка, полученная на лучшей модели
print(grid_cv.best_score_)


# In[22]:


# гиперпараметры лучшей модели
print(grid_cv.best_params_) 


# In[24]:


# создаем модель классификатора и учим его на имеющейся выборке
weigth_klass = SGDClassifier(alpha=6.785714285714286e-05, max_iter=8, learning_rate='optimal', eta0=1e-05)
weigth_klass.fit(arr, y = df_train.sex.values)


# In[25]:


# загрузим тест
df_test = pd.read_csv('df_test.csv')


# In[27]:


# сформируем датасет для решения задачи предсказания пола по весу и росту
df_test_cut = df_test[['growth', 'weight', 'sex']]


# In[28]:


# удалим из теста пропущенные значения
df_test_cut.dropna(inplace = True)


# In[29]:


# аналогично трейну, нормируем данные
arr_test = scaler.transform(df_test_cut[['growth', 'weight']].values.reshape(-1,2))


# In[30]:


# прогоняем тест через предикт
df_test_cut['predicted'] = weigth_klass.predict(arr_test)


# **Оценим полученные результаты**

# In[31]:


# загрузим метрики качества классификации
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[32]:


# процент правильно классифицированных объектов обоих классов
print(accuracy_score(df_test_cut.predicted, df_test_cut.sex))


# In[33]:


# доля верно классифицируемых объектов среди всех объектов, отнесенных к данному классу (точность)
# сколько объектов действительно относятся к классу среди тех, кого в этот класс определил классификатор
print(precision_score(df_test_cut.predicted, df_test_cut.sex, average = None, zero_division = 1))


# In[34]:


#  доля правильно классифицированных объектов соответствующего класса среди всех объектов этого класса (полнота)
print(recall_score(df_test_cut.predicted, df_test_cut.sex, average = None, zero_division = 1))


# In[35]:


# построим матрицу сопряженности
pd.crosstab(df_test_cut.predicted, df_test_cut.sex)


# In[36]:


# закодируем полученные результаты, пусть 0 - это верное предсказание,
# 1 - мужской пол определен как женский
# 2 - женский пол определен как мужской
df_test_cut['code'] = 0
df_test_cut.loc[(df_test_cut.sex == 'мужской')&(df_test_cut.predicted == 'женский'), 'code'] = '1'
df_test_cut.loc[(df_test_cut.sex == 'женский')&(df_test_cut.predicted == 'мужской'), 'code'] = '2'


# In[37]:


sns.scatterplot(data = df_test_cut, x = 'weight', y = 'growth', hue = 'code')


# In[40]:


# две самых высоких женщины (с весом 69 - 70 кг) были определены как мужчины
df_test_cut.query('code == "2"')


# In[41]:


# самые миниатюрные мужчины, схожие по комплекции с крупными женщинами из данной выборки, были ошибочно отнесены к классу женщин
df_test_cut.query('code == "1"')

