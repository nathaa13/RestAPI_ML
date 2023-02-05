from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse, request
import pickle
import numpy as np
import json
from tensorflow import keras
import numpy as np


app = Flask(__name__)
api = Api(app)



# Define how the api will respond to the post requests
class classify(Resource):
    def post(self):

        jsonfile= json.loads(request.data)
        X = (np.array(jsonfile['data'])/255).reshape([-1,28,28,1])
        prediction = model.predict(X)
        predicted_label=np.argmax(prediction)
        class_names = [ "T-shirt/top" , "Trouser" , "Pullover" , "Dress" , "Coat"
,"Sandal" , "Shirt" , "Sneaker" , "Bag" , "Ankle boot"]
        return jsonify(class_names[predicted_label])



api.add_resource(classify, '/classify')

if __name__ == '__main__':
    # Load model
    model = keras.models.load_model('model.h5')


    app.run(debug=True)



