
import string
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk .corpus import stopwords
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()


def transform_text(text):
  text= text.lower()
  text= nltk.word_tokenize(text)
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)
  text= y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  text= y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))
  return " ".join(y)
