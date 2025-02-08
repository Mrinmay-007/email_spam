

from flask import Flask, render_template, request
import pickle
from preprocess import transform_text  # Import text preprocessing function

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        message = request.form["message"]  # Get input text
        trans_sms = transform_text(message)  # Preprocess text
        vect_input = tfidf.transform([trans_sms])  # Vectorize
        prediction = model.predict(vect_input)[0]  # Predict

        result = "Spam" if prediction == 1 else "Not Spam"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run()



# # if button clicked 

# # 1. preprocess
# trans_sms = transform_text(text)

# # 2. vectorize 
# vect_input= tfidf.transform([trans_sms])

# # 3. predict
# result = model.predict(vect_input)[0]

# # 4. display
# if result == 1 :
#     print("spam")
# else:
# #     print("Not Spam")


