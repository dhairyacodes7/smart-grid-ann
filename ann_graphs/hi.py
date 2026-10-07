from numpy.testing import verbose
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
np.random.seed(42)
samples = 1000
dataset =[]
for i in range(samples):
    hours = np.random.randint(0,24)
    day = np.random.randint(1,8)
    temperature = np.random.uniform(20,40)
    previous_load = np.random.uniform(100,200)
    #create load
    load =(0.6*previous_load+2*hours+1.5*temperature+3*day+np.random.normal(0,5.))
    dataset.append([hours,day,temperature,previous_load,load])
#CREATE Dataframe
df = pd.DataFrame(dataset,columns=["Hours","Day","Temperature","Previous Load","Load"])
print("\nFirst Five records:")
print(df.head())
# SAVE DATASET
df.to_csv("electricity_dataset.csv",index=False)
print(f"\nDataset has been saved to electricity_dataset.csv {len(df)} records")
#visualizing data

plt.figure(figsize=(12,8))
plt.plot(df["Load"].values[:100],marker = "o",markersize = 3)
plt.title("Electrical load over time")
plt.xlabel("Samples")
plt.ylabel("Load(KW)")
plt.grid(True)
plt.savefig('plot1_load_over_time.png', dpi=150, bbox_inches='tight')
print('Saved: plot1_load_over_time.png')
plt.close()
#select 
X = df[['Hours','Day','Temperature','Previous Load']].values
y = df['Load'].values
#SPLIT DATA
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("\nTraining Samples:",len(X_train))
print("Testing Samples:",len(X_test))
#normalize input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
#create ann model
model = Sequential()
#input layer
model.add(Input(shape=(4,)))
#first hidden layer
model.add(Dense(64, activation="relu"))
#second hidden layer
model.add(Dense(32,activation="relu"))
#third hidden layer
model.add(Dense(16,activation="relu"))
#fourth hidden layer
model.add(Dense(8,activation="relu"))
#output layer
model.add(Dense(1))
#Compile Model
model.compile(optimizer="adam",loss="mse",metrics=["mae"])
#display model
print("\nANN Model Structure:")
model.summary()
#train model
print("\nTraining the model...")
history = model.fit(
    X_train,y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    verbose=1
)
print("\nTraining completed")
#plot trainig lossn
plt.figure(figsize=(12,6))
plt.plot(history.history['loss'],label="Train Loss",color='blue',marker="o",markersize=3)
plt.plot(history.history['val_loss'],label="Validation loss",color='red',marker="o",markersize=3)
plt.title("Training and Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig('plot2_training_loss.png', dpi=150, bbox_inches='tight')
print('Saved: plot2_training_loss.png')
plt.close()
#predict load
y_pred = model.predict(X_test,verbose =0)
#convert output into one dimensional array
y_pred = y_pred.flatten()
#calculate performance
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)
print("\n ----------")
print("ANN LOAD PREDICTION ")
print("-------------")
print("Mean Absolute Error (MAE):",mae)
print("Mean squared Error (MSE):",mse)
print("Root Mean Squared Error (RMSE):",rmse)
print("R2 Score:",r2)
#plot predicted vs actual load
plt.figure(figsize=(12,6))
plt.plot(y_test[:50],marker ="o",markersize=4,label="Actual Load")
plt.plot(y_pred[:50],marker ="s",markersize=4,label="Predicted Load")
plt.title("Actual vs Predicted load (First 50 samples)")
plt.xlabel("Samples")
plt.ylabel("Load (KW)")
plt.legend()
plt.grid(True)
plt.savefig('plot3_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
print('Saved: plot3_actual_vs_predicted.png')
plt.close()
#save the trained model
model.save("ann_electricity_model.h5")
print("\nModel saved as ann_electricity_model.h5")
#save scaler
import pickle
with open("scaler.pkl","wb") as f:
    pickle.dump(scaler,f)
print("Scaler saved as scaler.pkl")
# Display results



