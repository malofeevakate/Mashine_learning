#!/usr/bin/env python
# coding: utf-8

# Имеются результаты опроса студентов ТомГУ. Доступна информация:  
# 1. о физических параметрах опрашиваемых (возраст, рост, вес, пол, длина волос, размер обуви, год и месяц рождения, число удаленных зубов, наличие очков, разница в мм между средним и мизинцем/указательным/безымянными пальцами)  
# 2. о социальных параметрах (наличие сиблингов, количество друзей, расстояние до дома от универа, военнообязанность, номер этажа, размер родного города,  strange_people, social_network_duration_min, hostel) 
# 3. о предпочтениях (желаемое количество детей, любимые  животные, любимый фастфуд и шоколад, на что ставка в монетку и в игре "камень - ножницы - бумага")
# 4. об учебе (номер курса, факультет, оценки за ЕГЭ - при их наличии - по русскому, математике, физике, компьютерным наукам, химии, литературе, истории, географии, биологии, иностранному языку, социальным наукам, оценка в универе, наличие сложностей в учебе в последнем семестре, опоздания на первую пару в минутах, место в аудитории)  
# 5. остальные параметры (width_of_5000_mm, height_of_5000_mm)  
# 
# Попробуем решить следующие задачи с помощью метода случайного леса:  
# 1. Классификация объектов по полу, в зависимости от физических параметров, ну и добавим в рассмотрение желаемое число детей   
# 2. Задача предсказания института, где учится студент, по баллам, на которые он написал ЕГЭ

# In[33]:


# грузим либы
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# считаем данные и выделим нужные для задачи признаки
df_train = pd.read_csv('df_train.csv')
df_train = df_train[['growth', 'weight', 'sex', 'hair_length', 'children_number']]


# In[4]:


# поглядим на парные распределения наших фичей, в принципе женщины от мужчин в общем отличаются, что логично
sns.pairplot(df_train, hue = 'sex')


# In[10]:


# строим модель, максимальную глубину деревьев леса возьмем 2
model = RandomForestClassifier(max_depth = 2, random_state = 0)
model.fit(df_train[['growth', 'weight', 'hair_length', 'children_number']].values.reshape(-1, 4), y = df_train.sex.values)


# In[12]:


# грузим тест
df_test = pd.read_csv('df_test.csv')
# сформируем датасет для решения задачи предсказания пола по весу и росту
df_test = df_test[['growth', 'weight', 'sex', 'hair_length', 'children_number']]
# удаляем пустые значения, они бесполезны
df_test.dropna(inplace = True)
# формируем предикт
df_test['predicted'] = model.predict(df_test[['growth', 'weight', 'hair_length', 'children_number']].values.reshape(-1,4))
pd.crosstab(df_test.predicted, df_test.sex)


# In[13]:


# закодируем полученные результаты, пусть 0 - это верное предсказание,
# 1 - мужской пол определен как женский
# 2 - женский пол определен как мужской
df_test['code'] = 0
df_test.loc[(df_test.sex == 'мужской')&(df_test.predicted == 'женский'), 'code'] = '1'
df_test.loc[(df_test.sex == 'женский')&(df_test.predicted == 'мужской'), 'code'] = '2'


# In[18]:


# мужчина, классифицированный как женщина, имеет вес 50 кг и длинные волосы, рост средний для мужчин
# женщина же наоборот крупная, хоть и с длинными волосами
df_test.query('code != 0')


# In[19]:


# оценим метрики качества
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(df_test.predicted, df_test.sex)


# ***Попробуем использовать вероятностный лес для определения пола по ранее рассмотренным признакам***

# In[20]:


# строим модель, тренируем
model_probability = RandomForestClassifier(max_depth = 2, random_state = 0)
model_probability.fit(df_train[['growth', 'weight', 'hair_length', 'children_number']].values.reshape(-1, 4), y = df_train.sex.values)


# In[22]:


# грузим тест
df_test = pd.read_csv('df_test.csv')
# сформируем датасет для решения задачи предсказания пола по весу и росту
df_test = df_test[['growth', 'weight', 'sex', 'hair_length', 'children_number']]
# удаляем пустые значения, они бесполезны
df_test.dropna(inplace = True)


# In[23]:


# формируем предикт с вероятностями
result = model_probability.predict_proba(df_test[['growth', 'weight', 'hair_length', 'children_number']].values.reshape(-1,4))
print(result)


# In[24]:


# по вероятностям присваиваем классы
df_test['predicted_class_0'] = result[:, 0]
df_test['predicted_class_1'] = result[:, 1]


# In[25]:


df_test.head()


# In[26]:


# поищем ошибочно классифицированных мужчин (тот же, что и в прошлый раз)
df_test.loc[(df_test.sex == 'мужской')&(df_test.predicted_class_1 < 0.5)]


# In[27]:


# поищем ошибочно классифицированных женщин (та же женщина)
df_test.loc[(df_test.sex == 'женский')&(df_test.predicted_class_0 < 0.5)]


# ***Задача предсказания института, где учится студент, по баллам, на которые он написал ЕГЭ***

# In[28]:


# загрузим трейн
df = pd.read_csv('df_train.csv')


# In[29]:


# строим модель, тренируем
model_ins = RandomForestClassifier(max_depth = 4, random_state = 0)
model_ins.fit(df.iloc[:,6:17].values.reshape(-1, 11), y = df.your_insitute)


# In[30]:


# грузим тест
df_test = pd.read_csv('df_test.csv')


# In[31]:


# формируем предикт
df_test['predicted'] = model_ins.predict(df_test.iloc[:,6:17].values.reshape(-1, 11))


# In[32]:


pd.crosstab(df_test.predicted, df_test.your_insitute)


# In[38]:


# загрузим метрики качества классификации
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[39]:


# процент правильно классифицированных объектов
print(accuracy_score(df_test.predicted, df_test.your_insitute))


# In[40]:


df_institute = pd.pivot_table(df_test, index='your_insitute', values=['russian_rating', 'maths_rating', 'physics_rating',
       'computer_science_rating', 'chemistry_rating', 'literature_rating',
       'history_rating', 'geography_rating', 'biology_rating',
       'foreign_language_rating', 'social_science_rating'], aggfunc=np.mean)


# In[41]:


print(df_institute.idxmax())


# Хотя максимальные средние баллы по предметам совпадают с профилем соответствующих факультетов, модель наименее полно определяет экономистов, юристов и педагогов, которых не выделила вобще (видимо из-за пересечения сдаваемых предметов с другими факультетами, у экономистов это математики, у юристов - соцгум, педагоги тоже сдают неспецифические предметы с хорошими баллами), полнее всего - технарей и соцгум. 

# ***Попробуем найти наиболее значимые фичи в определении института***

# In[43]:


# загрузим основной датасет
df = pd.read_csv('df_train.csv')


# In[44]:


# преобразуем все категориальные признаки датасета в количественные
coder = preprocessing.LabelEncoder()
for name in df.select_dtypes(include = ['object']).columns:
    coder.fit(df[name])
    df[name] = coder.transform(df[name])


# In[45]:


# оценим значимость каждого признака в определении пола объекта
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
# определяем селектор
selector = ExtraTreesClassifier()
# определяем, относительно какого признака расчитываем значимость ив каком датасете
result = selector.fit(df[df.columns], df.your_insitute)
result.feature_importances_


# In[46]:


# создаем таблицу значимости
features_table = pd.DataFrame(result.feature_importances_, index = df.columns, columns = ['importance'])


# In[47]:


features_table.sort_values(by = ['importance'], ascending = False).head(10)

