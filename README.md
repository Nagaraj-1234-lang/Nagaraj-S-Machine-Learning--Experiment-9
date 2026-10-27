# Nagaraj-S-Machine-Learning--Experiment-9
### DATE : 3/11/2026






### AIM : 







Train a small feedforward neural network to classify non-linear data. 








### ALGORITHM : 









Two-layer dense network with ReLU hidden units and sigmoid output, trained with Adam. 








### THEORY:







With non-linear activations, networks approximate complex decision boundaries 
(universal approximation).





### INPUT : 
```
#EXPREIMENT 9
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x,y = make_moons(n_samples=500, noise=0.2, random_state=42)
xtr,xte,ytr,yte = train_test_split(x,y,test_size=0.2,random_state=42)
model = Sequential([Dense(8, activation='relu', input_shape=(2,)), Dense(1, activation='sigmoid')])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(xtr, ytr, epochs=30, batch_size=16, verbose=0)
y_pred = (model.predict(xte) >0.5).astype(int).flatten()
print('NN test accuracy:', round(accuracy_score(yte, y_pred),4))
```
### OUTPUT :
```

NN test accuracy: 0.87

```








### RESULT : 







Neural networks model non-linear boundaries; students see improved accuracy over 
linear methods.
