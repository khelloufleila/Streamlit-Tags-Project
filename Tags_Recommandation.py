import streamlit as st

import numpy as np

# NLP
import neattext.functions as nfx
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
# Text Downloader

import base64
import time
from PIL import Image
# EDA
import re
import pandas as pd 
import nltk
nltk.download('stopwords')
from wordcloud import WordCloud
from nltk.corpus import stopwords
import spacy
import os
import pickle
from spacy.tokenizer import Tokenizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.tokenize import ToktokTokenizer
from string import punctuation
token = ToktokTokenizer()
punct = punctuation


# Load NLP packages
import spacy
from spacy import displacy
nlp= spacy.load('en_core_web_sm')
#import en_core_web_sm
#nlp = en_core_web_sm.load()


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


top_tags = ['javascript','python','java','android','git','c#','c++','html','ios','css','jquery','.net','php','c','string','node.js','bash','sql','objective-c','mysql']

# Load the matrices 

LR = pd.read_pickle(r'LR.pkl')
x =  pd.read_pickle(r'x.pkl')
#y = pd.read_pickle(r'y.pkl')
data_tags= pd.read_pickle(r'data_tags.pkl')


multilabel_binarizer = MultiLabelBinarizer()
y = multilabel_binarizer.fit_transform(data_tags)

def avg_jaccard(y_true, y_pred):
    ''' It calculates Jaccard similarity coefficient score for each instance,and
    it finds their average in percentage

    Parameters:

    y_true: truth labels
    y_pred: predicted labels
    '''
    jacard = np.minimum(y_true, y_pred).sum(axis=1) / \
        np.maximum(y_true, y_pred).sum(axis=1)

    return jacard.mean()*100


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

def remove_contract_form(text):
    document = nltk.word_tokenize(text) 
    words = []
    for token in document:
        text = token
        text = re.sub(r"\'m", "am", text)
        text = re.sub(r"\'re", "are", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"n't", "not", text)
        text = re.sub(r"\'ve", "have", text)
        text = re.sub(r"\'d", "would", text)
        text = re.sub(r"\'ll", "will", text)
        words.append(text)
    return ' '.join(words)  

def delete_multiple_space(text):
    return ' '.join(text.split())  
# Delete string of length = 1 and not in tags
def low_length(text):
    document = nlp(text)
    words = []
    for token in document:
        if len(token.text) > 1:
            words.append(token.text)
        else:
            if token.text in top:
                words.append(token.text) 
    return ' '.join(words)  

def strip_list_noempty(mylist):

    newlist = (item.strip() if hasattr(item, 'strip')
               else item for item in mylist)

    return [item for item in newlist if item != '']
def clean_punct(text):
    ''' Remove all the punctuation from text, unless it's part of an important
    tag (ex: c++, c#, etc)

    Parameter:

    text: text to remove punctuation from it
    '''

    words = token.tokenize(text)
    punctuation_filtered = []
    regex = re.compile('[%s]' % re.escape(punct))
    remove_punctuation = str.maketrans(' ', ' ', punct)

    for w in words:
        if w in top_tags:
            punctuation_filtered.append(w)
        else:
            w = re.sub('^[0-9]*', " ", w)
            punctuation_filtered.append(regex.sub('', w))

    filtered_list = strip_list_noempty(punctuation_filtered)

    return ' '.join(map(str, filtered_list))   

timestr= time.strftime("%Y%m%d- %H%M%S")
def text_downloader(raw_text):
	b64= base64.b64encode(raw_text.encode()).decode()
	new_filename= "Clean_text_result_{}_txt".format(timestr)
	st.markdown("### Download file ###")
	href= f'<a href= "data: file/txt;base64, {b64}" download= "{new_filename}">click here!!</a>'
	st.markdown(href, unsafe_allow_html=True)



# Function to Analyse Tokens and Lemma
def text_analyzer(my_text):
	docx = nlp(my_text)
	# tokens = [ token.text for token in docx]
	allData = [(token.text ,token.pos_, token.lemma_, token.tag_, token.is_alpha) for token in docx ]
	df= pd.DataFrame(allData, columns=['Token','Pos', 'Lemma', 'Tag', 'IsAlpha'])

	return df


def Plot_worldcloud(my_text):
	my_worldcloud= WordCloud().generate(my_text)
	fig= plt.figure()
	plt.title('Top words for Our Text')
	plt.imshow(my_worldcloud,  interpolation="bilinear")
	plt.axis=('off')
	st.pyplot(fig)




def main():
	html_temp = """
	<div style="background-color:blue;padding:10px">
	<h1 style="color:white;text-align:center;">Recommandation Tags Application </h1>
	</div>
	"""
	st.markdown(html_temp,unsafe_allow_html=True)
	st.info("Leila Khellouf Project")
	st.subheader('StackOverflow dataset: https://data.stackexchange.com/stackoverflow/query/new')


	menu= ["TextCleaner",  "About"]
	choice= st.sidebar.selectbox("Menu", menu)

	if choice=="TextCleaner":
		st.subheader("Text Cleaning")
		text_file= st.file_uploader("upload txt file", type=['txt'])
		Cleaner_Text= st.sidebar.checkbox("Cleaner Text")

		if text_file is not None:
			# Decode Text
			raw_text= text_file.read().decode('utf-8')
			file_details= {"Filename": text_file.name, "FileSize": text_file.size, "FileType": text_file.type}
			st.write(file_details)
			col1, col2= st.beta_columns(2)
				

			with col1:
				with st.beta_expander("Original Text"):
					st.write(raw_text)

			with col2:
				with st.beta_expander("Processed Text"):
					if Cleaner_Text:
						#raw_text= raw_text.lower()
						
						raw_text= clean_text(raw_text)
						raw_text= BeautifulSoup(raw_text).get_text()
						raw_text= clean_punct(raw_text)
						raw_text= delete_multiple_space(raw_text)
						raw_text= remove_stopwords(raw_text)
						raw_text= remove_contract_form(raw_text)
						raw_text= stemm_text(raw_text)

				


					st.write(raw_text)
					text_downloader(raw_text)

				    

			with st.beta_expander("Text Analysis"):
				token_result_df = text_analyzer(raw_text)
				st.dataframe(token_result_df)

			with st.beta_expander("Plot WorldCloud"):
			     Plot_worldcloud(raw_text)
			
  
			with st.beta_expander("Modelisation "):
				
				min_df = st.slider('what should be the min_df of the TfidfVectorizer ?', min_value=0.001, max_value=0.1,  step=0.001)
				max_df = st.slider('what should be the max_df of the TfidfVectorizer ?', min_value=0.1, max_value=1.0, step=0.1)
			

				vectorizerIdf = TfidfVectorizer(analyzer='word', min_df=min_df, max_df=max_df, strip_accents=None,
				                                 encoding='utf-8', preprocessor=None, token_pattern=r'(?u)\b\w+\b',
				                                # token_pattern=r"(?u)\S\S+",
				                                 max_features=2000) 
				X_IDF= vectorizerIdf.fit_transform(x)
				X_idf = X_IDF.astype('float32')

				# Split our data 
				X_train, X_test, y_train, y_test = train_test_split(X_idf, y, test_size = 0.2, random_state = 0)

                # Fit the model 
				LR = OneVsRestClassifier(LogisticRegression(solver='liblinear', random_state=42, C=10.0, penalty='l1'))
				LR.fit(X_train, y_train)
				y_pred = LR.predict(X_test)
				jaccard = avg_jaccard(y_test, y_pred)


				st.subheader('Jaccard score in percentage for OVR with LogisticRegression')
				st.write(' %.2f' % jaccard,'%')
				#print('Jaccard score in percentage for OVR with LogisticRegression: %.2f' % jaccard)
				Cleaned_txt=[raw_text]

				TFIDF_question= vectorizerIdf.transform(Cleaned_txt)
				y_pred_encod= LR.predict(TFIDF_question)

				tags_encod= multilabel_binarizer.inverse_transform(y_pred_encod)

				st.subheader('The Recommandation Tags are: ')
				st.write(tags_encod)


			   
	else:
		st.subheader("About")	

			
					




if __name__ == '__main__':
	main()
