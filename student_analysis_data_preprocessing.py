#!/usr/bin/env python
# coding: utf-8

# Имеются результаты опроса студентов ТомГУ. Доступна информация:  
# 1. о физических параметрах опрашиваемых (возраст, рост, вес, пол, длина волос, размер обуви, год и месяц рождения, число удаленных зубов, наличие очков, разница в мм между средним и мизинцем/указательным/безымянными пальцами)  
# 2. о социальных параметрах (наличие сиблингов, количество друзей, расстояние до дома от универа, военнообязанность, номер этажа, размер родного города,  strange_people, social_network_duration_min, hostel) 
# 3. о предпочтениях (желаемое количество детей, любимые  животные, любимый фастфуд и шоколад, на что ставка в монетку и в игре "камень - ножницы - бумага")
# 4. об учебе (номер курса, факультет, оценки за ЕГЭ - при их наличии - по русскому, математике, физике, компьютерным наукам, химии, литературе, истории, географии, биологии, иностранному языку, социальным наукам, оценка в универе, наличие сложностей в учебе в последнем семестре, опоздания на первую пару в минутах, место в аудитории)  
# 5. остальные параметры (width_of_5000_mm, height_of_5000_mm)  
# 
# Проведем EDA (в т.ч. проверим наличие пропусков, взаимосязь в данных), преобразуем данные при необходимости

# In[3]:


# грузим либы
import pandas as pd
import seaborn as sns


# In[25]:


# считаем данные
df = pd.read_csv('students.csv')
# изменим имена колонок на удобные
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.lower()
df.info()


# Видно, что в некоторых столбцах имеются пропуски, а именно:  
#     - желаемое число детей,  
#     - число удаленных зубов,  
#     - вес,  
#     - носит ли человек очки  
# Для того, чтобы понять, что делать с пропусками, поисследуем соответствующие показатели:  
# 1. поделим группы по полу,  
# 2. изучим распределения данных,  
# 4. заменим пропущенные значения признака либо на среднее, либо на медиану (поскольку показатели количественные)

# In[26]:


# начнем с женщин, выделим их в отдельный датасет
df_fem = df.query('sex == "женский"')
sns.boxplot(data = df_fem.weight)


# In[29]:


df_fem.isna().sum()


# In[27]:


# посмотрим на распределение веса в группе
df_fem.weight.hist()


# Итак, мы имеем:  
# - 1 случай указания веса менее 20 кг, что вряд ли является истинным, так как по остальному сету минимум начинается от примерно 40 кг  
# - 25 пропущенных значений веса
# 
# *Предлагается*:  
# 1. пустые значения веса определить как средние по полу (поскольку распределение достаточно симметрично и унимодально)  
# 2. удалить наблюдения со значением веса менее 20 кг  

# In[28]:


df_fem = df_fem.query('(weight > 20) | (weight != weight)')
df_fem.weight = df_fem.weight.fillna(df_fem.weight.mean())


# In[29]:


# а что у женщин с количеством желаемых детей
df_fem.children_number.hist()


# In[36]:


df_fem.children_number.describe()


# In[30]:


# распределение полимодально, большинство опрашиваемых хотят 1-2 детей, заменим пустоты на медиану
df_fem.children_number = df_fem.children_number.fillna(1.5)


# In[7]:


# а что с потерянными зубами?
df_fem.removed_teeth.hist()


# In[31]:


df_fem.removed_teeth.median()


# In[32]:


# распределение полимодально, большинство опрашиваемых зубов не теряли, заменим пустоты на медиану
df_fem.removed_teeth = df_fem.removed_teeth.fillna(0)


# In[33]:


# выделим группу мужчин в отдельный датасет
df_man = df.query('sex == "мужской"')


# In[53]:


df_man.isna().sum()


# In[54]:


# посмотрим на распределение веса в группе
df_man.weight.hist()


# Итак, мы имеем:  
# - 4 пропущенных значения веса,  
# - 1 случай указания веса менее 20 кг, что вряд ли является истинным, так как по остальному сету минимум начинается от примерно 40 кг  
# 
# *Предлагается*:  
# 1. пустые значения веса определить как средние по полу (поскольку распределение достаточно симметрично и унимодально)  
# 2. удалить наблюдения со значением веса менее 20 кг  

# In[34]:


df_man = df_man.query('(weight > 20) | (weight != weight)')
df_man.weight = df_man.weight.fillna(df_man.weight.mean())


# In[10]:


# а что у мужчин с количеством желаемых детей
df_man.children_number.hist()


# In[60]:


df_man.children_number.describe()


# In[35]:


# в данных есть выброс, который есть смысл не учитывать при поиске медианы:
df_man.query('children_number < 1000').children_number.describe()


# In[36]:


# удалять ничего не будем, но заменим пустые значения на медиану без учета одного ответа (1024 ребенка)
df_man.children_number = df_man.children_number.fillna(1.5)


# In[65]:


# посмотрим на распределение потерянных зубов
df_man.removed_teeth.hist()


# In[37]:


# распределение полимодально, большинство опрашиваемых зубов не теряли, заменим пустоты на медиану
df_man.removed_teeth = df_man.removed_teeth.fillna(0)


# In[14]:


# также 2 мужчины не указали, носят ли они очки
df_man.glasses.mode()


# In[38]:


# заменим пустоту по очкам на моду
df_man.glasses = df_man.glasses.fillna('нет')


# In[39]:


# объединим полученные датасеты
df = df_fem.append(df_man)


# In[1]:


# выгрузим полученный датасет для дальнейшей работы
file_name = 'df_train.csv'
df.to_csv(file_name, index = False)


# In[5]:


# грузим тест
df_test = pd.read_csv('students_test.csv')
# исправим написание колонок, уберем пробел и заменим прописные буквы на строчные
df_test.columns = df_test.columns.str.replace(' ', '_')
df_test.columns = df_test.columns.str.lower()
# удалим пустые значения, по ним ничего не предскажешь
df_test.dropna(inplace = True)


# In[6]:


# выгрузим полученный датасет для дальнейшей работы
file_name = 'df_test.csv'
df_test.to_csv(file_name, index = False)

