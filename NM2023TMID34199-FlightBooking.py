#!/usr/bin/env python
# coding: utf-8

# In[80]:


#Activity 1:Collect the dataset
#Activity 1.1:Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,confusion_matrix
import warnings
import pickle
from scipy import stats
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')


# In[81]:


#Activity 1.2:Reading the Dataset
import pandas as pd
df=pd.read_csv("E:\\NMDS\\FlightBooking.csv")
df.head()


# In[82]:


df.info()


# In[83]:


#import prision
#from prision import Category
#for i in category:
    #print(i,data[i].unique())


# In[84]:


#We now split the Date column to extract the 'Date','Month' and 'Year' values, and store them in 
#new columns in our data frame
df.date_of_Journey=df.Date_of_Journey.str.split('/')
df.Date_of_Journey


# In[85]:


#Traiting the data_column
df['Date']=df.Date_of_Journey.str[0]
df['Month']=df.Date_of_Journey.str[1]
df['Year']=df.Date_of_Journey.str[2]


# In[86]:


#Split the Route column
df.Total_Stops.unique()


# In[87]:


#We split the data in route column
df.Route=df.Route.astype(str).str.split('->')
df.Route


# In[88]:


df['City1']=df.Route.str[0]
df['City2']=df.Route.str[1]
df['City3']=df.Route.str[2]
df['City4']=df.Route.str[3]
df['City5']=df.Route.str[4]
df['City6']=df.Route.str[5]


# In[89]:


#In similar manner, we split the Dep_time column, and create separate for departue hours and minutes
df.Dep_Time=df.Dep_Time.astype(str).str.split(':')
df['Dep_Time_Hour']=df.Dep_Time.str[0]
df['Dep_Time_Mins']=df.Dep_Time.str[1]


# In[90]:


#We also split the Arrival_Time Column
df.Arrival_Time=df.Arrival_Time.astype(str).str.split('')
df['Arrival_date']=df.Arrival_Time.str[1]
df['Time_of_Arrival']=df.Arrival_Time.str[0]


# In[91]:


df['Time_of_Arrival']=df.Time_of_Arrival.astype(str).str.split(':')
df['Arrival_Time_Hour']=df.Time_of_Arrival.str[0]
df['Arrival_Time_Mins']=df.Time_of_Arrival.str[1]


# In[92]:


#we also treat the 'Total_stops', column, and replace non-stop flights with 0 values and
#extract the integer part of the "Total_Stops"
df.Total_Stops.replace('npn_stop',0,inplace=True)
df.Total_Stops=df.Total_Stops.str.split('')
df.Total_Stops=df.Total_Stops.str[0]


# In[93]:


df.Additional_Info.unique()


# In[94]:


#df.Additional_Info.replace('No Info','No Info',inplace=True)


# In[95]:


df.shape


# In[96]:


df.info()


# In[97]:


#we also drop some columns  like 'city6' an 'city5', since majority of the data in these columns was NaN(null)
#df.drop(['City4','City5','City6'],axis=1,inplace=True)
#df.drop(['Date_of_Journey','Route','Dep_Time','Duration'],axis=1,inplace=True)
#df.drop(['Time_of_Arrival'],axis=1,inplace=True)


# In[98]:


df.isnull().sum()


# In[99]:


#Activity 2.1:Replacing Missing Values
#filling City3 as None, the missing value are less
df['City3'].fillna('None,inplace=True')


# In[100]:


#filling Arrival_Date as Departure_Date
df['Arrival_date'].fillna((df['Date']),inplace=True)


# In[101]:


#filling Travel_Mins as Zero(0)
#df['Travel_Mins'].fillna(0,inplace=True)
df['Arrival_Time_Mins'].fillna(0,inplace=True)


# In[102]:


df.info()


# In[103]:


df.skew()


# In[104]:


#changing the numerical columns from object to int
#df.Total_Stops=df.Total_Stops.astype('int64')
df.Date=df.Date.astype('int64')
df.Month=df.Month.astype(str)
df.Year=df.Year.astype(str)
df.Dep_Time_Hour=df.Dep_Time_Hour.astype('int64')
df.Dep_Time_Hour=df.Dep_Time_Hour.astype('int64')
df.Dep_Time_Mins=df.Dep_Time_Mins.astype('int64')


# In[105]:


[df['Arrival_Time_Hour']=='5m']


# In[106]:


df.drop(index=6474,inplace=True,axis=1)


# In[107]:


#df.Travel_Hours=df.Travel_Hours.astype('int64')
#df.Arrival_Time_Hour=df.Arrival_Time_Hour.astype('int64')


# In[108]:


#Creating a list of Different types of Columns
Categorical=['Airline','Source','Destination','Additional_Info','City1']
Numerical=['Total_Stops','Date','Month','Year','Dep_Time_Hour','Dep_Time_Mins','Arrival_date',
           'Arrival_Time_Hour','Arrival_Time_Mins','Travel_Time_Mins','Travel_Hours','Travel_Mins']


# In[109]:


#Activity 2.2: Label Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df.Airline=le.fit_transform(df.Airline)
df.Source=le.fit_transform(df.Source)
df.Destination=le.fit_transform(df.Destination)
df.Total_Stops=le.fit_transform(df.Total_Stops)
df.City1=le.fit_transform(df.City1)
df.City2=le.fit_transform(df.City2)
df.City3=le.fit_transform(df.City3)
df.Additional_info=le.fit_transform(df.Additional_Info)
df.head()


# In[110]:


#Activity 2.3: Output Columns
df=df[['Airline','Source','Destination','Date','Month','Year','Dep_Time_Hour','Dep_Time_Mins','Arrival_date',
           'Arrival_Time_Hour','Arrival_Time_Mins','Price']]
df.head()


# In[111]:


#Activity 3:Exploratory Data Analyis
#Activity 3.1: Descriptive statistical
df.describe()


# In[112]:


#Ploting Countplots for Categorical Data
import matplotlib.pyplot as plt
import seaborn as sns
c=1
plt.figure(figsize=(20,45))
for i in Categorical:
    plt.subplot(6,3,c)
    sns.countplot(df[i])
    plt.xticks(rotation=90)
    plt.tight_layout(pad=3.0)
    c=c+1
    plt.show()


# In[114]:


#Distribution of price column
plt.figure(figsize=(15,8))
sns.distplot(df.Price)


# In[113]:


sns.heatmap(df.corr(),annot=True)


# In[115]:


#Detecting Outliers
import seaborn as sns 
sns.boxplot(df['Price'])


# In[ ]:


#Scaling the data
df=df[['Airline','Source','Destination','Date','Month','Year','Dep_Time_Hour','Dep_Time_Mins','Arrival_date',
           'Arrival_Time_Hour','Arrival_Time_Mins','Price']]
y=df['Price']
x=df.drop(columns=['Price'],axis=1)
import pandas as pd
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
#x_scaled=ss.fit_transform(x)
#x_scaled=pd.DataFrame(x_scaled,columns=x.columns)
x.head()


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[ ]:


x_train.tail()


# In[ ]:


#Model Building 
#model 1:RandomForestClassifier,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingRegressor,AdaBoostRegressor
rfr=RandomForestClassifier()
gb=GradientBoostingRegressor()
ad=AdaBoostRegressor()


# In[ ]:


from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import r2_score,mean_absoulte_error,mean_squared_error
for i in [rfr,gb,ad]:
    i.fit(x_train,y_train)
    y_pred=i.predict(x_test)
test_score=r2_score(y_test,y_pred)
train_score=r2_score(y_train,i.predict(x_train))
if abs(train_score-test_score)<0.2:
    print(i)
    print("R2 score is",r2_score(y_test,y_pred))
    print("R2 for train data",r2_score(y_train,i.predict(x_train)))
    print("Mean Absoult Error is",mean_absolute_error(y_pred,y_test))
    print("Mean Squared Error is",mean_squared_error(y_pred,y_test))
    print("Root Mean Squared Error is",(mean_squared_error(y_pred,y_test,squared=False)))


# In[ ]:


#model 2:RandomForestClassifier,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from.sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_absoulte_error,mean_squared_error
knn=KNeighborsRegressor()
svr=SVR()
dt=DecisionTreeRegressor()
for i in [knn,svr,dt]:
    i.fit(x_train,y_train)
    y_pred=i.predict(x_test)
test_score=r2_score(y_test,y_pred)
train_score=r2_score(y_train,i.predict(x_train))
if abs(train_score-test_score)<0.1:
    print(i)
    print("R2 score is",r2_score(y_test,y_pred))
    print("R2 for train data",r2_score(y_train,i.predict(x_train)))
    print("Mean Absoult Error is",mean_absolute_error(y_pred,y_test))
    print("Mean Squared Error is",mean_squared_error(y_pred,y_test))
    print("Root Mean Squared Error is",(mean_squared_error(y_pred,y_test,squared=False)))



# In[ ]:


#model 3: Checking Cross Validation for RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingRegressor,AdaBoostRegressor
rfr=RandomForestClassifier()
gb=GradientBoostingRegressor()
ad=AdaBoostRegressor()
for i in [rfr,gb,ad]:
    i.fit(x_train,y_train)
    y_pred=i.predict(x_test)
for i in range(2,5):
    CV=cross_val_score(rfr,x,y,CV=i)
print(rfr,CV.mean())

