import pandas as pd 
import numpy as np
import streamlit as st
import re
import nltk

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfTransformer

vectorizerIdf = TfidfVectorizer(analyzer='word',
                                    min_df=0.01,
                                    max_df=0.7,
                                    strip_accents=None,
                                    encoding='utf-8',
                                    preprocessor=None,
                                    token_pattern=r'(?u)\b\w+\b',
                                    # token_pattern=r"(?u)\S\S+",
                                    max_features=2000) 

header= st.beta_container()
dataset= st.beta_container()
features= st.beta_container()
model_training= st.beta_container()



# TODO: Change values below and observer the changes in your app
st.markdown(
	"""
	<style>
	.main{
	background-color: #F5F5F5;
	}
	""",
	unsafe_allow_html=True
	)

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 90%;
        padding-top: 5rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 5rem;
    }}
    img{{
    	max-width:40%;
    	margin-bottom:40px;
    }}
</style>
""",
        unsafe_allow_html=True,
    )

def clean_text(text):
    # Make text lowercase
    text = str(text).lower()
    # remove text in square brackets,
    text = re.sub('\[.*?\]', '', text)
    # remove links
    text = re.sub('https?://\S+|www\.\S+', '', text)
    # remove punctuation rajouter les if pour ne pas supprimer les .net et c++!!!!!!!!!!!!
    #text = re.sub('<*?>', '', text )
    # remove words containing numbers.
    #text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

stop_words= stopwords.words('english')
more_stopwords= ['don', 'im', 'o']
stop_words= stop_words + more_stopwords

def remove_stopwords(text):
    text= ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text   

stemmer = nltk.SnowballStemmer("english")

def stemm_text(text):
    text= ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text


def Recommanded_tags(question):
	question= clean_text(question)
	question= BeautifulSoup(question).get_text()
	question= clean_punct(question)
	question= remove_stopwords(question)
	question= remove_contract_form(question)
	question= stemm_text(question)
	Cleaned_question=[question]

	TFIDF_question= vectorizerIdf.transform(Cleaned_question)
	y_pred_encod= LR.predict(TFIDF_question)

	tags_encod= multilabel_binarizer.inverse_transform(y_pred_encod)
	return tags_encod





with header:
	st.title('Welcome to my awesome NLP project!')
	st.text('In this project I look into Tags recommondation!')

with dataset:
	st.header('StackOverflow dataset')
	st.text('I found this dataset on **stackexchange explore**.!')


with features:
	st.header('The features I created')

	st.markdown('* **TF feature:** *i created this feature because of this ......')
	st.markdown('* **TF_IDF feature:** *i created this feature because of this ......')
	st.markdown('* **Glove feature:** *i created this feature because of this ......')





with model_training:

	st.header('Time to train the model!')
	st.text('Here you get to choose the hyperparameters of the model and the performance chanse')


	sel_col, disp_col = st.beta_columns(2)



#question= input('Ask your question: ')
#tags_encod= Recommanded_tags(question)
#print('Recommended tags are: {}'.format(tags_encod))


