from  flask import Flask 
# Creation of the Flask app
app = Flask(__name__)

# API 
# Flask route so that we can serve HTTP traffic on that route
@app.route('/',methods=['POST', 'GET'])
# Return predictions of inference using Iris Test Data
def prediction():

    # Load and split the data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)

    # Classification score
    clf = load(MODEL_PATH)
    score = clf.score(X_test, y_test)

    return {'score': score}

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080) # Launch built-in we server and run this Flask webapp
