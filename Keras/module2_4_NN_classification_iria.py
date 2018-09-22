# Step 1

import pandas as pd
iris = pd.read_csv('iris.csv')
X = iris.drop('species',axis=1).values
y = iris[['species']].values

import numpy as np
num_labels = len(np.unique(y))
all_Y = np.eye(num_labels)[y]  # One liner trick!
all_Y= np.squeeze(all_Y)

# from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,all_Y,test_size=0.25,random_state=25)


# Step 2: Build the Model
import keras
from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(20, input_dim=4, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Step 3: Compile the Model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Step 4: Train the Model
model.fit(X_train,y_train,epochs=200,verbose=0)

# Step 5: Evaluate the Model
loss,accuracy = model.evaluate(X_test,y_test)
print("Loss = ",loss)
print("Accuracy ",accuracy)