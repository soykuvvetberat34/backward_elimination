import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
datas=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Makine Öğrenmesi\\çoklu doğrusal regresyon\\deneme.csv")

# Eksik verilerin olduğu yerlere sütundaki tüm verilerin ortalamasını koyma
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
age = datas.iloc[:, 1:4].values
imputer = imputer.fit(age)  # Değerleri eğitir
age = imputer.transform(age)  # Değerleri değiştirir
age = age.astype(int)  # Tüm yaş değerlerini int yapar
print(age)
age_df = pd.DataFrame(age, columns=["boy", "kilo", "yaş"])

# Ülke isimleri için encoding işlemi
countrys = datas.iloc[:, 0].values.reshape(-1, 1)
# One-Hot encoding
OHE = OneHotEncoder()
OHE_Countrys = OHE.fit_transform(countrys).toarray()
OHE_Countrys = OHE_Countrys.astype(int)
df_countrys = pd.DataFrame(data=OHE_Countrys, columns=OHE.categories_[0])

# Cinsiyet değerleri için encoding işlemi
genders = datas.iloc[:, -1].values.reshape(-1, 1).ravel()#reshape(-1, 1) ifadesi tek boyutlu diziyi iki boyutlu hale getirir. Ancak LabelEncoder kullanırken .ravel() kullanarak verileri tek boyutlu diziye dönüştürmek gereklidir.
# Label encoding
LE_gender = LabelEncoder()
genders_LE = LE_gender.fit_transform(genders)
gender_df = pd.DataFrame(genders_LE, columns=["genders"])

# Verileri birleştirme
res1 = pd.concat([df_countrys, age_df], axis=1)
res2 = pd.concat([res1, gender_df], axis=1)

boy=datas.iloc[:,1].values.reshape(-1,1)

#backward elemination
res3=res2.iloc[:,[0,1,2,3,4]].values
array=np.append(arr=np.ones((20,1)).astype(int),values=res3,axis=1)
arr_l=np.array(array,dtype=float)
model=sm.OLS(boy,arr_l).fit()
print(model.summary())# bu kod blogundaki p değeri büyük olan değişkenleri sildik

res4=res2.iloc[:,[0,2,3]].values
array=np.append(arr=np.ones((20,1)).astype(int),values=res4,axis=1)
arr_l=np.array(array,dtype=float)
model=sm.OLS(boy,arr_l).fit()
print(model.summary())#bu bloguda inceleyip p değeri büyük olan değişkenleri sildik

res5=res2.iloc[:,[2,3]].values
array=np.append(arr=np.ones((20,1)).astype(int),values=res5,axis=1)
arr_l=np.array(array,dtype=float)
model=sm.OLS(boy,arr_l).fit()
print(model.summary())

x_train,x_test,y_train,y_test=train_test_split(res5,boy,test_size=0.33,random_state=1)

regression=LinearRegression()
regression.fit(x_train,y_train)
predict_reg=regression.predict(x_test)
print(x_test)
print(predict_reg)










