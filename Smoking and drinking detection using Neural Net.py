#!/usr/bin/env python
# coding: utf-8

# In[2]:


#basic imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df.head()


# In[22]:


df.info()


# In[6]:


df.describe().T


# In[3]:


from ydata_profiling import ProfileReport
rep = ProfileReport(df)
rep


# In[4]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['DRK_YN'] = le.fit_transform(df['DRK_YN'])
df['sex'] = le.fit_transform(df['sex'])


# In[5]:


df.drop_duplicates(inplace=True)


# In[28]:


get_ipython().run_line_magic('matplotlib', 'inline')
df.select_dtypes(include='number').corr()['DRK_YN'].drop('DRK_YN').sort_values(ascending=False).plot(kind='bar')


# In[30]:


sns.heatmap(df.select_dtypes(include='number').corr(),cmap='magma',linecolor='white',linewidths=1)


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('white')
sns.countplot(x='DRK_YN',data=df,palette='viridis')


# In[17]:


facet_kws = {'hue':'DRK_YN','palette':{0:'red',1:'blue'}}
sns.displot(x='height',data=df,col='DRK_YN',**facet_kws,bins=40)


# In[18]:


sns.countplot(x='SMK_stat_type_cd',data=df,palette='viridis',hue='DRK_YN')
#if SMK_stat_type_cd is < 1 more cases of class 1(Drunk)


# In[29]:


facet_kws = {'hue':'DRK_YN','palette':{0:'red',1:'blue'}}
sns.displot(x='age',data=df,col='DRK_YN',**facet_kws,bins=20,alpha=0.6)


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.kdeplot(df.weight)


# # defining a function to draw boxplots

# In[26]:


def plot_boxplots(df):
    cols = df.select_dtypes(include='number').columns
    num_plots = len(cols)
    num_rows = (num_plots+1)//2
    fig , axes = plt.subplots(nrows=num_rows,ncols=2,figsize=(20,20))
    for i,column in enumerate(cols):
        col = i %2
        row = i//2
        ax = axes[row,col]
        sns.boxplot(x=df[column],ax=ax)
        ax.set_title(f"Boxplot of {column}")
        ax.set_xlabel(column)
    plt.tight_layout()
    plt.show()


# In[27]:


plot_boxplots(df)


# # detecting outliers using IQR (skewed data)

# In[6]:


def remove_outliers(df,columns,k=1.5):
    "function to remove outliers using IQR method"
    #calculate q1(first quartile) and q3(third quartile)
    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        #calculate inter-quartile range 
        iqr = q3-q1
        #remove values which lie below q1-1.5(iqr) and above q3+1.5(iqr)
        df[column] = df[column].clip(lower = q1 - k*iqr, upper = q3 + k+iqr)
    return df


# In[7]:


df = remove_outliers(df,['waistline','SBP', 'DBP','BLDS','tot_chole','triglyceride','serum_creatinine','SGOT_AST', 'SGOT_ALT','sight_left',
       'sight_right', 'hear_left', 'hear_right'],k=1.5)


# ## Neural Network

# In[11]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


# In[25]:


# 24 -> 12 -> 6 ->3 ->1
model = Sequential()
#layer 1 (input)
model.add(Dense(24,activation='relu'))
model.add(Dropout(0.2))
#layer 2
model.add(Dense(12,activation='relu'))
model.add(Dropout(0.2))
#layer 3 
model.add(Dense(6,activation='relu'))
model.add(Dropout(0.2))
#layer 4
model.add(Dense(3,activation='relu'))
model.add(Dropout(0.2))
#output
model.add(Dense(1,activation='sigmoid'))

#compiling
model.compile(loss='binary_crossentropy',optimizer='adam')


# In[26]:


model.fit(X_train,y_train,
          validation_data=(X_val,y_val),
          batch_size=1024,
         epochs = 50)


# In[17]:


from tensorflow.keras.models import load_model
model.save('smoking_class_project_model.keras') 


# In[27]:


#plotting loss
loss_df = pd.DataFrame(model.history.history)


# In[15]:


loss_df.head()


# In[28]:


loss_df.plot()


# In[18]:


predictions = (model.predict(X_test) > 0.5).astype("int32")


# In[29]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))

