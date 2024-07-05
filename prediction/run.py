from flask import Flask 
from flask_restx import Api, Resource 

# server for additional models

app = Flask(__name__)  
api = Api(app)

@api.route('/predict')  
class HelloWorld(Resource):
    def __init__(self):
        self.model # load model

    def get(self):
        
        return {"hello": "world!"}

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)