#!/usr/bin/env python
# coding: utf-8

# # DZ 1 Немного EDA для маркетинговых данных

# *Вам предложены данные iFood — приложения для доставки еды в Бразилии, представленного более чем в тысяче городов. Поддержание высокой вовлеченности клиентов является важной задачей для компании. Для того, чтобы создавать новые предложения и проводить качественные рекламные кампании, необходимо понимать особенности и потребности целевой аудитории. Для этого были собраны данные о клиентах, использующих приложение. Вам предстоит выявить особенности людей, пользующихся данным приложением и найти интересные закономерности.*

# *Описание данных:*
# 
# * AcceptedCmp1 - 1 if customer accepted the offer in the 1st campaign, 0 otherwise
# * AcceptedCmp2 - 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
# * AcceptedCmp3 - 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
# * AcceptedCmp4 - 1 if customer accepted the offer in the 4th campaign, 0 otherwise
# * AcceptedCmp5 - 1 if customer accepted the offer in the 5th campaign, 0 otherwise
# * Response (target) - 1 if customer accepted the offer in the last campaign, 0 otherwise
# * Complain - 1 if customer complained in the last 2 years
# * DtCustomer - date of customer’s enrolment with the company
# * Education - customer’s level of education
# * Marital - customer’s marital status
# * Kidhome - number of small children in customer’s household
# * Teenhome - number of teenagers in customer’s household
# * Income - customer’s yearly household income
# * MntFishProducts - amount spent on fish products in the last 2 years
# * MntMeatProducts - amount spent on meat products in the last 2 years
# * MntFruits - amount spent on fruits products in the last 2 years
# * MntSweetProducts - amount spent on sweet products in the last 2 years
# * MntWines - amount spent on wine products in the last 2 years
# * MntGoldProds - amount spent on gold products in the last 2 years
# * NumDealsPurchases - number of purchases made with discount
# * NumCatalogPurchases - number of purchases made using catalogue
# * NumStorePurchases - number of purchases made directly in stores
# * NumWebPurchases - number of purchases made through company’s web site
# * NumWebVisitsMonth - number of visits to company’s web site in the last month
# * Recency - number of days since the last purchase

# ### Задание 1 (1 балл): предобработка данных

# In[73]:


import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("marketing.csv")
df.info()


# *В представленных данных есть два типа данных- количественные и категориальные и данные.Признак Dt_Customer представлен в неудобном виде.*

# In[46]:


df['Income'] = pd.to_numeric(df['Income'],errors = 'coerce')


# In[54]:


df['Dt_Customer']=pd.to_datetime(df['Dt_Customer'], errors='ignore')


# ### Задание 2 (1.5 балла): пропуски и выбросы

# **Обработка пропущенных значений**

# *В начале нужно убедиться, есть ли пропущенные значения в исходных данных, и удалим признаки с пропущенными значениями, если их больше 95%, так как он считаются неинформативными:*

# In[57]:


for col in df.columns:
    missing=np.mean(df[col].isnull())
    print("{}-{}%".format(col, round(missing*100)))


# In[60]:


plt.figure(figsize=(10, 7))
df.isnull().sum().plot(kind="bar", color="tomato");


# *Оставшиеся признаки с пропущенными значениями заменю на статистические велечины, так как это один из самых надежных способов.*

# In[64]:


df.Income.fillna(df.Income.mode()[0], inplace=True)


# **Удаление аномалий**

# *Теперь стоит избавиться от выбросов в данных, для этого всем выбросам дадим значние nan и заменим на медианное значение*

# In[14]:


df_nomic=data.select_dtypes(include=[np.number])
nomic_cols=df_nomic.columns.values
for col in nomic_cols:
    missing=data[col].isnull()
    df_missing=np.sum(missing)
    if df_missing>0:
        print('inpytying missing value for:{}'.format(col))
        med=data[col].median()
        data[col]=data[col].fillna(med)


# ### Задание 3 (1 балл): новые признаки

# *Можно разделить дату на три признака, дата , день и месяц.Тем самым мы упростим наши данные.*

# In[8]:


df['Day']=df['Dt_Customer'].dt.weekday


# In[9]:


df['Month']=df['Dt_Customer'].dt.month


# In[10]:


df['Year']=df['Dt_Customer'].dt.year


# ### Задание 4 (2.5 балла): время статистики!

# In[70]:


df.describe()


# *По представленным данным можн сделать вывод, что в данных после обработки выбросов практически нет, так как медианное и среднее значение практически не отличаются.Дисперсия большая присутствует только в трех признаках, это тоже хорошо, так как данные очень хорошо обучаются с низкой дисперсией.*

# ### Задание 5 (1 балл): корреляции

# *Чтобы посчитать взаимосвязи между переменными, можно использовать коэффициент корреляции Пирсона. Это мера интенсивности и направления линейной зависимости между двумя переменными. Значение +1 означает идеально линейную положительную зависимость, а -1 означает идеально линейную отрицательную зависимость. Хотя этот коэффициент не может отражать нелинейные зависимости, с него можно начать оценку взаимосвязей переменных. В Pandas можно легко вычислить корреляции между любыми колонками в кадре данных (dataframe):*

# In[97]:


# Select the numeric columns
numeric_subset = df.select_dtypes('number')

# Create columns with square root and log of numeric columns
for col in numeric_subset.columns:
    # Skip the Energy Star Score column
    if col == 'AcceptedCmp4':
        next
    else:
        numeric_subset['sqrt_' + col] = np.sqrt(numeric_subset[col])
        numeric_subset['log_' + col] = np.log(numeric_subset[col])

# Select the categorical columns
categorical_subset = df[['AcceptedCmp3', 'AcceptedCmp5']]

# One hot encode
categorical_subset = pd.get_dummies(categorical_subset)

# Join the two dataframes using concat
# Make sure to use axis = 1 to perform a column bind
features = pd.concat([numeric_subset, categorical_subset], axis = 1)

# Drop buildings without an energy star score
features = features.dropna(subset = ['AcceptedCmp4'])

# Find correlations with the score 
correlations = features.corr()['AcceptedCmp4'].dropna().sort_values()


# In[98]:


correlations.head(3)


# ### Задание 6 (2 балла) : визуализации

# In[71]:


df.hist(figsize=(15,10),color="c",bins=50);


# *На графике видно, что данные расположены относительно неплохо, так как практически все они семмитричны(большое занчение находится в середине, а меньшие по бокам)*

# In[74]:


sns.pairplot(df);


# *Видно, что в выборке нет выбросов, а также , большинство признаков не имеют схожести, это озночает, что такие признаки можно оставить*

# In[85]:


sns.boxplot(data=df, palette='Spectral');


# *На этои графике видно, что большинство значений имеет низкое среднее значение, и как ранее было замечено выбросов в данных не присутствует.*

# In[87]:


sns.distplot(df['Year_Birth']);


# *На этомграфике видно распределения признака Year_Birth, он также хорошо располжен и семетричен.*

# ### Задание 7 (1 балл): немного исследований

# In[100]:


df[(df["Marital_Status"] =="Together") & (df["Country"] =="AUS")]


# *Мы определили, что с материальным статусом Together в стране AUS существует 41 человек*
