import streamlit as st
import pickle
import string
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download the 'punkt' tokenizer data
nltk.download('punkt')

# Download the 'stopwords' data
nltk.download('stopwords')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email Spam Classifier")

input_mail = st.text_area("Enter the mail")

if st.button('Predict'):

    # 1. preprocessing
    transformed_mail = transform_text(input_mail)
    # 2. vectorization
    vector_input = tfidf.transform([transformed_mail])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("The above mail is spam")
    else:
        st.header("The above mail is not spam")
