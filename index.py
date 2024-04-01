from itertools import combinations, combinations_with_replacement
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Pre_processing import *
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#Load data
data = pd.read_csv('assignment2dataset.csv')

#Variables
degrees = 3
iterations = 1000
learningRate = 0.0001

#Preprocessing
data.dropna(how='any',inplace=True)
performanceData = data.iloc[:,:]

#Encoding
lbl = LabelEncoder()
lbl.fit(data['Extracurricular Activities'].values)
data['Extracurricular Activities'] = lbl.transform((data['Extracurricular Activities'].values))


X = data.iloc[:,0:6] #Features 
Y = data['Performance Index'] #Target



#Feature Selection
corr = performanceData.corr(numeric_only=True)
top_feature = corr.index[abs(corr['Performance Index'])>0.3]

#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = performanceData[top_feature].corr()
sns.heatmap(top_corr, annot=True)
#plt.show()
top_feature = top_feature.delete(-1)
X = X[top_feature]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30,shuffle=True,random_state=10)

#Scaling
scale = StandardScaler()
X_train['Hours Studied']= scale.fit_transform(X_train.iloc[:, 0:1])
X_train['Previous Scores']= scale.fit_transform(X_train.iloc[:, 1:2])
#y_train['Performance Index'] = scale.fit_transform(y_train.iloc[:,-1:])

X_test['Hours Studied']= scale.fit_transform(X_test.iloc[:, 0:1])
X_test['Previous Scores']= scale.fit_transform(X_test.iloc[:, 1:2])
#y_test['Performance Index'] = scale.fit_transform(y_test.iloc[:, -1:])


#polynomaial regression (training data)
X_poly = pd.DataFrame()
c = X_train.shape[1]
n = X_train.shape[0]
res = 1

for i in range(degrees):
    lst = list(combinations_with_replacement(range(X_train.shape[1]), i+1))
    for l in lst:
        for c in l:
            res = X_train.iloc[:,[c]].to_numpy()*res

        res = pd.DataFrame(res)
        X_poly = pd.concat([X_poly, res], axis=1)
        res = 1

X_poly.insert(0, '0', np.power(X_train['Hours Studied'].values,0)*0)
X_poly.columns = X_poly.columns.astype(str) #عشان تكون أسماء الأعمدة كلها من نفس التايب

print("X_poly")
print(X_poly)

#polynomaial regression (Training Data)
X_test_poly = pd.DataFrame()
res = 1

for i in range(degrees):
    lst = list(combinations_with_replacement(range(X_test.shape[1]), i+1))
    for l in lst:
        for c in l:
            res = X_test.iloc[:,[c]].to_numpy()*res
        res = pd.DataFrame(res)
        X_test_poly = pd.concat([X_test_poly, res], axis=1)
        res = 1

X_test_poly.insert(0, '0', np.power(X_test['Hours Studied'].values,0)*0)
X_test_poly.columns = X_test_poly.columns.astype(str) 
print(X_test_poly)


poly_model = linear_model.LinearRegression()
poly_model.fit(X_poly, y_train)
prediction = poly_model.predict(X_test_poly)


print('Co-efficient of linear regression',poly_model.coef_)
print('Intercept of linear regression model',poly_model.intercept_)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
