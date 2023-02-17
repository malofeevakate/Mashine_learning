#!/usr/bin/env python
# coding: utf-8

# Имеются результаты опроса студентов ТомГУ. Доступна информация:  
# 1. о физических параметрах опрашиваемых (возраст, рост, вес, пол, длина волос, размер обуви, год и месяц рождения, число удаленных зубов, наличие очков, разница в мм между средним и мизинцем/указательным/безымянными пальцами)  
# 2. о социальных параметрах (наличие сиблингов, количество друзей, расстояние до дома от универа, военнообязанность, номер этажа, размер родного города,  strange_people, social_network_duration_min, hostel) 
# 3. о предпочтениях (желаемое количество детей, любимые  животные, любимый фастфуд и шоколад, на что ставка в монетку и в игре "камень - ножницы - бумага")
# 4. об учебе (номер курса, факультет, оценки за ЕГЭ - при их наличии - по русскому, математике, физике, компьютерным наукам, химии, литературе, истории, географии, биологии, иностранному языку, социальным наукам, оценка в универе, наличие сложностей в учебе в последнем семестре, опоздания на первую пару в минутах, место в аудитории)  
# 5. остальные параметры (width_of_5000_mm, height_of_5000_mm)  
# 
# Попробуем решить следующие задачи с помощью метода градиентного бустинга:  
# 1. Классификация объектов по полу, в зависимости от физических параметров, ну и добавим в рассмотрение желаемое число детей   
# 2. Задача предсказания института, где учится студент, по баллам, на которые он написал ЕГЭ

# In[26]:


import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# считаем данные и выделим нужные для задачи признаки
df_train = pd.read_csv('df_train.csv')
df_train = df_train[['growth', 'weight', 'sex', 'hair_length', 'children_number']]


# In[3]:


sns.pairplot(df_train, hue = 'sex')


# In[4]:


# строим модель, тренируем
model = GradientBoostingClassifier(random_state = 0)
model.fit(df_train[['growth', 'weight', 'hair_length', 'children_number']].values.reshape(-1, 4), y = df_train.sex.values)


# In[5]:


# грузим тест
df_test = pd.read_csv('df_test.csv')
# сформируем датасет для решения задачи предсказания пола по весу и росту
df_test = df_test[['growth', 'weight', 'sex', 'hair_length', 'children_number']]
# удаляем пустые значения, они бесполезны
df_test.dropna(inplace = True)
# формируем предикт
df_test['predicted'] = model.predict(df_test[['growth', 'weight', 'hair_length', 'children_number']].values.reshape(-1,4))
pd.crosstab(df_test.predicted, df_test.sex)


# In[6]:


# оценим метрики качества
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(df_test.predicted, df_test.sex)


# In[14]:


df_test.query('(predicted == "женский") and (sex == "мужской")')


# В целом, бустинг справился лучше леса: крупную женщину он верно определил, ошибся с невысоким мужчиной с длинными волосами

# ***Задача предсказания института, где учится студент, по баллам, на которые он написал ЕГЭ***

# In[15]:


# считаем данные
df_train = pd.read_csv('df_train.csv')


# In[17]:


# строим модель, тренируем
model_ins = GradientBoostingClassifier(random_state = 0)
model_ins.fit(df_train.iloc[:,6:17].values.reshape(-1, 11), y = df_train.your_insitute)


# In[18]:


# грузим тест
df_test = pd.read_csv('df_test.csv')
# проверим пропущенные значения
df_test.iloc[:,6:17].info()


# In[19]:


#формируем предикт
df_test['predicted'] = model_ins.predict(df_test.iloc[:,6:17].values.reshape(-1, 11))


# In[20]:


pd.crosstab(df_test.predicted, df_test.your_insitute)


# In[22]:


print(accuracy_score(df_test.predicted, df_test.your_insitute))


# In[23]:


print(precision_score(df_test.predicted, df_test.your_insitute, average = None, zero_division = 1))


# In[24]:


print(recall_score(df_test.predicted, df_test.your_insitute, average = None, zero_division = 1))

