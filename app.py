from flask import Flask
import nltk

app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def home():

    fidubil = request.get("fi")
    if fidubil:
        return "Bora fiii du Billl " + fidubil
        
    return "Booora Biiiill!!!"
    
if __name__ == "__main__":
    app.run(debug=True)