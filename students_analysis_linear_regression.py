#!/usr/bin/env python
# coding: utf-8

# Имеются результаты опроса студентов ТомГУ. Доступна информация:  
# 1. о физических параметрах опрашиваемых (возраст, рост, вес, пол, длина волос, размер обуви, год и месяц рождения, число удаленных зубов, наличие очков, разница в мм между средним и мизинцем/указательным/безымянными пальцами)  
# 2. о социальных параметрах (наличие сиблингов, количество друзей, расстояние до дома от универа, военнообязанность, номер этажа, размер родного города,  strange_people, social_network_duration_min, hostel) 
# 3. о предпочтениях (желаемое количество детей, любимые  животные, любимый фастфуд и шоколад, на что ставка в монетку и в игре "камень - ножницы - бумага")
# 4. об учебе (номер курса, факультет, оценки за ЕГЭ - при их наличии - по русскому, математике, физике, компьютерным наукам, химии, литературе, истории, географии, биологии, иностранному языку, социальным наукам, оценка в универе, наличие сложностей в учебе в последнем семестре, опоздания на первую пару в минутах, место в аудитории)  
# 5. остальные параметры (width_of_5000_mm, height_of_5000_mm)  
# 
# Посмотрим, насколько хорошо можно решить задачу предсказания роста опрашиваемых по другим физическим показателям методом линейной регрессии

# In[32]:


# грузим либы
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

get_ipython().run_line_magic('matplotlib', 'inline')


# In[39]:


# считаем трейн
df_train = pd.read_csv('df_train.csv')


# А какие количественные признаки вобще в бОльшей степени влияют на рост? Скорее всего, это признаки, относящиеся к физическим (age, shoe_size, removed_teeth, weight, month_of_birthday, hair_length, middle_and_ring_finger, middle_and_little_finger, middle_and_index_finger).

# In[40]:


# урежем трейн до количественных физических признаков, посмотрим что там
df_train = df_train [['growth','age', 'shoe_size', 'removed_teeth', 'weight', 'month_of_birthday', 'hair_length',
                      'middle_and_ring_finger', 'middle_and_little_finger', 'middle_and_index_finger']]


# In[41]:


# определяем селектор
selector = ExtraTreesClassifier()
# определяем, относительно какого признака расчитываем значимость ив каком датасете
result = selector.fit(df_train[df_train.columns], df_train.growth)
result.feature_importances_


# In[42]:


# создаем таблицу значимости
features_table = pd.DataFrame(result.feature_importances_, index = df_train.columns, columns = ['importance'])


# In[43]:


# сильно выделяющихся значимостей нет, возьмем все
features_table.sort_values(by = ['importance'], ascending = False)


# In[44]:


# смотрим на распределение данных - кажется, линейная модель должна неплохо справиться с предсказанием
sns.pairplot(data = df_train)


# In[48]:


# values переводит серию в массив чисел np.ndarray (в т.ч. целевые)
# reshape транспонирует указанный массив чисел, превращая массив в вектор чисел типа np.ndarray
X_train = df_train[['age', 'shoe_size', 'removed_teeth', 'weight', 'month_of_birthday', 'hair_length',
                    'middle_and_ring_finger', 'middle_and_little_finger', 'middle_and_index_finger']].values.reshape(-1,9)


# In[50]:


y_train = df_train.growth.values


# In[49]:


# функция оценки модели
def get_cv_scores(model):
    scores = cross_val_score(model,
                             X_train,
                             y_train,
                             cv=5,
                             scoring='r2')
    
    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))
    print('\n')


# In[51]:


# создаем модель линейной регрессии МНК
# обучаем fit(значения НЕЦЕЛЕВЫХ признаков, y = значение ЦЕЛЕВОГО признака)
linRegr_growth = LinearRegression()
results = linRegr_growth.fit(X_train, y_train)


# In[52]:


# только 46% дисперсии мы можем объяснить полученной моделью
get_cv_scores(linRegr_growth)


# Мы получаем значение R² 0,46 и стандартное отклонение 0,13. Низкое значение R² указывает на то, что наша модель не очень точна. Значение стандартного отклонения указывает на то, что мы можем ее переобучить.

# In[53]:


# считаем тест
df_test = pd.read_csv('df_test.csv')
# оставим нужные для модели параметры
df_test_cut = df_test[['growth','age', 'shoe_size', 'removed_teeth', 'weight', 'month_of_birthday', 'hair_length',
                      'middle_and_ring_finger', 'middle_and_little_finger', 'middle_and_index_finger']]


# In[54]:


X_test = df_test[['age', 'shoe_size', 'removed_teeth', 'weight', 'month_of_birthday', 'hair_length',
                    'middle_and_ring_finger', 'middle_and_little_finger', 'middle_and_index_finger']].values.reshape(-1,9)


# In[57]:


y_test = df_test.growth.values


# In[58]:


print('Train Score: ', linRegr_growth.score(X_train, y_train))
print('Test Score: ', linRegr_growth.score(X_test, y_test))


# In[59]:


# посмотрим на коэффициенты
linRegr_growth.intercept_


# In[60]:


linRegr_growth.coef_


# In[61]:


df_test['pred_growth'] = results.predict(X_test)


# In[62]:


# считаем МSЕ теста
mean_squared_error(df_test.growth, df_test.pred_growth)


# Попробуем улучшить модель с использованием ***Ridge - регрессии***

# In[63]:


from sklearn.linear_model import Ridge

# тренируем модель с исходным альфа = 1
ridge = Ridge(alpha=1).fit(X_train, y_train)

# посмотрим на оценки модели
get_cv_scores(ridge)


# In[64]:


print('Train Score: ', ridge.score(X_train, y_train))
print('Test Score: ', ridge.score(X_test, y_test))


# In[65]:


# поищем оптимальуню альфу
alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = dict(alpha=alpha)

grid = GridSearchCV(estimator=ridge, param_grid=param_grid, scoring='r2', verbose=1, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)


# In[66]:


ridge = Ridge(alpha=0.01).fit(X_train, y_train)

get_cv_scores(ridge)

print('Train Score: ', ridge.score(X_train, y_train))
print('Test Score: ', ridge.score(X_test, y_test))


# Практически без изменений.  
# Попробуем улучшить модель с использованием ***Lasso - регрессии***  
# 
# Лассо-регрессия использует регуляризацию L1, чтобы некоторые коэффициенты были точно равны нулю. Это означает, что некоторые функции полностью игнорируются моделью. Это можно рассматривать как тип автоматического выбора функций

# In[67]:


from sklearn.linear_model import Lasso

# аналогично риджу, учим на стандартной альфа = 1
lasso = Lasso(alpha=1).fit(X_train, y_train)

# оцениваем
get_cv_scores(lasso)


# In[68]:


print('Train Score: ', lasso.score(X_train, y_train))
print('Test Score: ', lasso.score(X_test, y_test))


# In[69]:


# поищем оптимальную альфу
alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = dict(alpha=alpha)

grid = GridSearchCV(estimator=lasso, param_grid=param_grid, scoring='r2', verbose=1, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)


# In[70]:


lasso = Lasso(alpha=1).fit(X_train, y_train)

get_cv_scores(lasso)

print('Train Score: ', lasso.score(X_train, y_train))
print('Test Score: ', lasso.score(X_test, y_test))


# In[71]:


lasso.intercept_


# In[72]:


# отметим, что лассо учитывает не все параметры, 4 из 9 обнуляет
lasso.coef_


# Практически без изменений.  
# Попробуем улучшить модель с использованием ***Эластичной сети***  
# 
# Эластичная сеть — это модель линейной регрессии, которая сочетает в себе штрафы Лассо и Риджа.
# 
# Мы используем этот l1_ratioпараметр для управления комбинацией регуляризации L1 и L2. Когда l1_ratio = 0у нас есть регуляризация L2 (Ridge) и когда l1_ratio = 1у нас есть регуляризация L1 (Lasso). Значения между нулем и единицей дают нам комбинацию регуляризации L1 и L2.

# In[75]:


from sklearn.linear_model import ElasticNet

# треним модель по дефолту
elastic_net = ElasticNet(alpha=1, l1_ratio=0.5).fit(X_train, y_train)

# оценим ее
get_cv_scores(elastic_net)


# In[76]:


# поищем оптимальную альфу
alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
l1_ratio = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
param_grid = dict(alpha=alpha, l1_ratio=l1_ratio)

grid = GridSearchCV(estimator=elastic_net, param_grid=param_grid, scoring='r2', verbose=1, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)


# Оптимальные коэффициенты эластичной сети совпадают с ***лассо - регрессией***  
# Проверим лассо - модель на тесте, оценим получившиеся результаты

# In[78]:


lasso = Lasso(alpha=1).fit(X_test, y_test)

get_cv_scores(lasso)


# Модель не очень справляется с предсказанием роста по остальным физическим признакам.
