import numpy as np
import requests
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import time
from tensorflow import keras


# Load data
fashion_test_df = pd.read_csv('fashion-mnist_test.csv')

testing = np.asarray(fashion_test_df, dtype='float32')
X_test = testing[:, 1:].reshape([-1,28,28,1])
X_test = X_test/255    # Normalizing the data

y_test = testing[:, 0]
        
class_names = [ "T-shirt/top" , "Trouser" , "Pullover" , "Dress" , "Coat"
,"Sandal" , "Shirt" , "Sneaker" , "Bag" , "Ankle boot"]

val=random.randint(0,len(X_test)-1)
# Serialize the data into json and send the request to the model

payload = {'data': (X_test[val].reshape(784)*255).tolist()}


y_predict = requests.post('http://127.0.0.1:5000/classify', json=payload)




print(payload)

# Load model
model = keras.models.load_model('model.h5')

#Evaluate the model


eval_result = model.evaluate(X_test, y_test)
print("Accuracy : {:.3f}".format(eval_result[1]))

print('Expected value:' + class_names[int(y_test[val])])
print('Prediction:' + y_predict.text)
