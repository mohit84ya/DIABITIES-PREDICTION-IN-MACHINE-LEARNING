#!/usr/bin/env python
# coding: utf-8

# # import library from python

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score 


# # Import File

# In[2]:


df= pd.read_csv(r"C:\Users\V Team\Downloads\diabetes.csv")


# In[3]:


df =pd.DataFrame(df)


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df.shape


# In[8]:


df


# # Statical method

# In[9]:


df.describe()


# In[10]:


df.info()


# In[11]:


df.isnull().sum()


# In[12]:


df['Outcome'].value_counts()

0 --> Non Diabitic
1 --> Diabitic
# In[13]:


df.groupby('Outcome').mean()


# # separating the data labels

# In[14]:


x= df.drop('Outcome', axis =1)
y =df['Outcome']


# In[15]:


y


# # Data Standarize 

# In[16]:


scaler =StandardScaler()


# In[17]:


standrize_data=scaler.fit_transform(x)


# In[18]:


x=standrize_data
y=df['Outcome']


# In[19]:


x,y


# # Train Test Split Model 

# In[20]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =.2,stratify=y, random_state =2)


# In[21]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# # Training The Model

# In[ ]:


classfier=svm.SVC(kernel="linear")


# In[ ]:


#train the support vector machine
classfier.fit(x_train,y_train)


# # Model Evaluation 

# # Accuracy Score

# In[ ]:


# Accuracy score of train data #
x_train_prediction=classfier.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction, y_train)
print('Accuracy_score = ',training_data_accuracy)


# In[ ]:


# Accuracy score of test data #
x_test_prediction=classfier.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction, y_test)
print('Accuracy_score of test data = ',test_data_accuracy)


# # Making a predective system

# In[ ]:


petient_report =int(input("enter your petient number"))

def  
for petient_report in df.Outcome:
    for df.Outcome[0] ==0:
        return('This petient Not diabitic')
        break
    else:
        return('This petient is diabitic')


# In[ ]:


df['Outcome']


# In[ ]:


sns.countplot(df['Outcome'])


# In[ ]:


df.columns


# In[ ]:


sns.jointplot(df['Outcome'],df['SkinThickness'],kind='reg')


# In[ ]:


plt.plot(df);


# In[ ]:




