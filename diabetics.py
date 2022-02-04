import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
data=pd.read_csv(r'C:\Users\moham\Desktop\diabetes.csv')
data.info()
data.describe()
data.isnull().sum()
 data.head()
x=np.arange(1,120)
y=2*x+5
plt.title("diabetics predictor")
plt.xlabel("bloodpressure")
plt.ylabel("age")
plt.plot(x,y)
plt.show()
x=data.iloc[:,:-1]
y=data.iloc[:,[-1]]
print(x,y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = .2,random_state = 0)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 6, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
y=pd.DataFrame(y_pred)
y.to_csv('out.csv',index=False,header=False)

