from flask import Flask
import nltk

app = Flask(__name__)

@app.route("/")
def home():
    return "Booora Biiiill!!!"
    
if __name__ == "__main__":
    app.run(debug=True)