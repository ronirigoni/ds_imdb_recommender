from flask import Flask, request
import nltk

app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def home():

    #if request.method == 'GET':

    fidubil = request.args.get("fi")
    if fidubil:
        return "Bora fiii du Billl " + fidubil
        
    return "Booora Biiiill!!!"
    
if __name__ == "__main__":
    app.run(debug=True)