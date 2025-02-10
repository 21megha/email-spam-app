import streamlit as st
import pickle 
import string
pip install nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('punkt')
# nltk.download('punkt_tab')
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


tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

st.title("Email Spam Classification Application")
st.write("This is a machine learning project by Megha Vishvkarma to detect spam or ham i.e. not spam")
    

user_input = st.text_area("Enter the SMS")

if st.button('Predict'):

  if user_input:
      data = [user_input]
      vectorized_data = tk.transform(data).toarray()
      result = model.predict(vectorized_data)
      if result[0]==0:
          st.write("The email is not spam")
      else:
          st.write("The email is spam")
  else:
      st.write("Please type Email to classify")
    
    
    
    # 1. preprocess
   # transformed_sms = transform_text(user_input)
    # 2. vectorize
   # vector_input = tk.transform([transformed_sms])
    # 3. predict
  #  result = model.predict(vector_input)[0]
    # 4. Display
  #  if result == 1:
   #     st.header("Spam")
  #  else:
   #     st.header("Not Spam")



# import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')

# import streamlit as st
# import pickle 
# import string
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer

# ps = PorterStemmer()


# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)


# tk = pickle.load(open("vectorizer.pkl", 'rb'))
# model = pickle.load(open("model.pkl", 'rb'))

# st.title("SMS Spam Detection Model")
# st.write("*Made by Edunet Foundation*")
    

# input_sms = st.text_input("Enter the SMS")

# if st.button('Predict'):

#     # 1. preprocess
#     transformed_sms = transform_text(input_sms)
#     # 2. vectorize
#     vector_input = tk.transform([transformed_sms])
#     # 3. predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")
