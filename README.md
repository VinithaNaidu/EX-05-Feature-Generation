# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
```
Name : D.Vinitha 
Reg No. : 212222230175
```
# Data.csv :
```
import pandas as pd
df=pd.read_csv("data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
```
# Encoding.csv :
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
oe=OrdinalEncoder()

df1["nom_0"] = oe.fit_transform(df1[["nom_0"]])
temp=['Cold','Warm','Hot']
oe2=OrdinalEncoder(categories=[temp])
df1['ord_2'] = oe2.fit_transform(df1[['ord_2']])

df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df0=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df2=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df3=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df4=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df4
```
# Titanic.csv :
```
import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

#removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

#data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

#feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
```
# OUPUT
# Data.csv :
# Initial Dataset:
![image](https://user-images.githubusercontent.com/121166004/232681919-12a12bbd-1349-41d6-af68-a10d86646efe.png)
# Binary Encoding:
![image](https://user-images.githubusercontent.com/121166004/232682122-74ab3bf3-a6c0-4fa4-bab4-de78e1b29aa0.png)
![image](https://user-images.githubusercontent.com/121166004/232682159-5cb4db07-dbf0-4c27-9032-90c4c1ffc7d0.png)
# Encoded Dataset:
![image](https://user-images.githubusercontent.com/121166004/232682320-0bc5dfa1-bc5d-48c6-a06c-d5a402e23fc7.png)
# Data Scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/121166004/232682444-0638474c-2c8f-4475-b051-daed9e874e84.png)
# Data Scaling using StandardScaler:
![image](https://user-images.githubusercontent.com/121166004/232682702-07276067-7658-45b4-bad9-11fae5ccddfe.png)
# Data Scaling using MaxAbsScaler:
![image](https://user-images.githubusercontent.com/121166004/232682832-6000dcec-7231-4b5d-8e95-b37032999de7.png)
# Data Scaling using RobustScaler:
![image](https://user-images.githubusercontent.com/121166004/232682936-fccd6acd-8da7-4fcc-b6e7-8f1082b2ba0e.png)
# Encoding.csv :
# Initial Dataset:
![image](https://user-images.githubusercontent.com/121166004/232683068-7253014c-af0c-46f2-97cb-435f278c9780.png)
# Binary Encoding:
![image](https://user-images.githubusercontent.com/121166004/232683162-89a62747-7a2e-4189-81fc-1c229b057bb3.png)
![image](https://user-images.githubusercontent.com/121166004/232683199-6f7eb9a2-cdf6-4d49-b3f7-e5d7e82517a3.png)
# Encoded Dataset:
![image](https://user-images.githubusercontent.com/121166004/232683429-859d2689-6642-4eac-9699-b8225537c3bd.png)
# Data Scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/121166004/232683487-6de20a87-63de-4324-95bb-17acb65f6401.png)
# Data Scaling using StandardScaler:
![image](https://user-images.githubusercontent.com/121166004/232683553-aba1ee6a-e671-438c-914a-9ef82d5a94a1.png)
# Data Scaling using MaxAbsScaler:
![image](https://user-images.githubusercontent.com/121166004/232683728-39abf68b-e13b-4fea-9ef4-ae363298ac5c.png)
# Data Scaling using RobustScaler:
![image](https://user-images.githubusercontent.com/121166004/232684128-cc59f3ad-74b9-458a-9d93-999e8edbff8a.png)
# Titanic.csv :
# Initial Dataset:
![image](https://user-images.githubusercontent.com/121166004/232684291-237f3ab2-03c7-4664-8a1b-8ed31293cc6d.png)
# Data cleaning before encoding:
# Cleaned Dataset:
![image](https://user-images.githubusercontent.com/121166004/232684798-c5dd397a-2ecb-49ef-a5a8-468b526b73ac.png)
![image](https://user-images.githubusercontent.com/121166004/232684944-0d36d38e-d74a-4b54-9f9d-198751416146.png)
![image](https://user-images.githubusercontent.com/121166004/232684985-67dfaade-31ab-4a7f-b45a-e0d9dbc2e2e6.png)
# Cleaned Dataset:
![image](https://user-images.githubusercontent.com/121166004/232685116-a5f86a92-3893-4b52-9d73-9460b63d66fc.png)
# Binary Encoding:
![image](https://user-images.githubusercontent.com/121166004/232685224-6fd99b62-e3d8-43d5-a4a1-42a7adbbc08a.png)
# Encoded Dataset:
![image](https://user-images.githubusercontent.com/121166004/232685591-c5227ca1-4374-4df4-9876-be7dff66e293.png)
# Data Scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/121166004/232685903-0f0a60b2-8f77-4277-999c-fe3a15ab6f4f.png)
# Data Scaling using StandardScaler:
![image](https://user-images.githubusercontent.com/121166004/232685964-3677fcd3-295e-4308-a649-7164e9714ef6.png)
# Data Scaling using MaxAbsScaler:
![image](https://user-images.githubusercontent.com/121166004/232686070-14be4e97-685a-4111-b49c-6aa78a104e8d.png)
# Data Scaling using RobustScaler:
![image](https://user-images.githubusercontent.com/121166004/232686174-6578bf49-1590-48c7-aeb9-c061bbbe50ab.png)
RESULT:
Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.

