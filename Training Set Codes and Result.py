
# coding: utf-8

# In[1]:

#Importing NLTK and downloading corpus for stop words
import nltk

nltk.download()


# In[2]:

import pandas


# In[3]:

#use pd.read_csv
messages = pandas.read_csv('Training_set.csv',
                           names=["label", "message"])
messages.head()


# In[4]:

#exploratory data analysis
messages.describe()


# In[5]:

#info() method
messages.info()


# In[6]:

#Use groupby to describe by label
#Grouping by the label column and then using the describe method
#as the aggregate function on the groupby method - for groupby, we always need to have an aggregate function
messages.groupby('label').describe()


# In[7]:

#As we continue our analysis we want to start thinking about the features we are going to be using. 
#This goes along with the general idea of feature engineering. 
#One feature to consider here is message length
messages['length'] = messages['message'].apply(len)
#I am only looking to apply the length function on the message column of the messages data frame -
messages.head()


# In[8]:

#Let's do some data visualization based on message length
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic(u'matplotlib inline')


# In[9]:

#Histogram of the length of the messages
#Call plot on the column of the pandas data frame
#plot is the more general method - 
#set the number of bins - either 50 or 100 (just play around and see what works better)
messages['length'].plot(bins=50,kind='hist')
#Question: why does the x-axis on the histogram plot - go all the way up to 800?
#Is there a really long message?


# In[10]:

#Find out through min, max and mean
messages['length'].max()


# In[11]:

messages['length'].min()


# In[12]:

messages['length'].mean()


# In[13]:

#Take a look at that huge message of 753 characters
messages[messages['length']==753]


# In[14]:

#Take a look at the entire message
#The iloc[0] command is just to print it out
messages[messages['length']==753]['message'].iloc[0]


# In[15]:

#Take a look at the tiny message of just 1 character
messages[messages['length']==1]


# In[16]:

#Is message length a distinguishing factor between the 11 categories of text messages?
#Generate 11 histograms
messages.hist(column='length', by='label', bins=50,figsize=(20,40))
#The x-axis measures the number of characters in a message for a specific category label
#The y-axis measures the frequency count of the messages belonging to that category label
#The label category greeting has the least number of characters, judging by the range of the x-axis 
#for this specific label category
#Label categories such as check-in, voucher and voucher not working even have messages consisting of
#characters ranging between 500 and 800 (judging by the range of the x-axis)


# In[17]:

#The messages in the data set are in the form of strings
#in text format - the classification algorithms need somesort of numerical feature 
#vector in order to perform the classification task
#The task here is to convert a corpus into a vector format
#corpus is a word for a group of texts
#Let's use the bag of words approach - where each unique word in the text has a unique number
#we will convert the raw messages (sequence of characters) into vectors (sequences of numbers).
#As a first step, let's write a function that will split a message into its individual words 
#and return a list. We'll also remove very common words, ('the', 'a', etc..). 
#To do this we will take advantage of the NLTK library.
#The First task is to remove punctuation. We can just take advantage of Python's built-in string 
#library to get a quick list of all the possible punctuation
import string


# In[18]:

#Let's create a function that will process the string in the message column, then we can 
#just use apply() in pandas do process all the text in the DataFrame.
#Our main goal is to create a function that will process a string in the message column
#so as to get Pandas to do all the processing of the words!
#Take these raw messages and then turn them into vectors - 
#take a sequence of characters and turn them into a sequence of numbers
#Create a variable called mess
mess = 'Sample message! Notice: it has punctuation.'
#First, remove the puncuation
#Instead of applying everything onto the data frame, we just mess around with the variable mess
#We can gather all the different transformations made on this and set it up as a function later on
#The following generates a string that contains puncuation marks
string.punctuation


# In[19]:

#Check characters to see if they are in punctuation
#nonpunc is a vector that is to contain all the non-puncuation marks in the
#form of capital letters and small letters - it replicates the message in mess 
#(in the form strings) without the puncuation
nopunc = [char for char in mess if char not in string.punctuation]


# In[20]:

nopunc


# In[21]:

# Join the strings again to form a large string.
nopunc = ''.join(nopunc)


# In[22]:

nopunc


# In[23]:

#Next task is to remove stop words
#Stopwords are common English words
#NLTK has the most support for English language
from nltk.corpus import stopwords


# In[24]:

#Let's see a preview of the object stopwords 
#in terms of the words that it contains
#Grab the 1st 10 words in stopwords
stopwords.words('english')[0:10]
#These stopwords are so common that they are not going to give us much information during classification
#Stopwords have an equally possible chance of occuring in each of the 11 text message label categories


# In[25]:

#split nopunc into separate words
nopunc.split()


# In[26]:

#Get rid of the stop words
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
#an object with the clean message (a message without the stop words)
#examine each word in the list called nopunc.split
#The above code is doing the following:
#It is considering the lower case instance of each word in nopunc.split
#if the word is not in the list of stop words, then include it in 
#the clean_mess list
clean_mess


# In[27]:

#Combine all these steps of text preprocessing and put them into a function
def text_process(mess):
    
    #Takes in a string of text, then performs the following:
    #1. Remove all punctuation
    #2. Remove all stopwords
    #3. Returns a list of the cleaned text
   
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[28]:

#Here is the original DataFrame again:
messages.head()


# In[29]:

#Need to tokenize the messages in the data frame
#It is the process of converting normal text strings 
#into a list of tokens -which are the words that we actually want
#Essentially, all we are doing is to apply the function text_process
#to the column message in the messages dataframe
#The code below is just to make sure that the function is working!
messages['message'].head(5).apply(text_process)


# In[30]:

#Continuing Normalization
#The task is to convert our list of words to an actual vector that SciKit-Learn can use.


# In[31]:

#Currently, we have the messages as lists of tokens (also known as lemmas) and now we need to convert 
#each of those messages into a vector the SciKit Learn's algorithm models can work with.

#Now we'll convert each message, represented as a list of tokens (lemmas) above, 
#into a vector that machine learning models can understand.

#We'll do that in three steps using the bag-of-words model:

#Count how many times does a word occur in each message (Known as term frequency)

#Weigh the counts, so that frequent tokens get lower weight (inverse document frequency)

#Normalize the vectors to unit length, to abstract from the original text length (L2 norm)


# In[32]:

#Let's begin the first step:

#Each vector will have as many dimensions as there are unique words in the SMS corpus. 
#We will first use SciKit Learn's CountVectorizer. 
#This model will convert a collection of text documents to a matrix of token counts.

#We can imagine this as a 2-Dimensional matrix. Where the 1-dimension is the entire vocabulary (each row represents every word
#available in the corpus) and the other dimension are the actual documents, in this case one column per text message.

#Since there are so many messages, we can expect a lot of zero counts for the presence of that word in that document. 
#Because of this, SciKit Learn will output a Sparse Matrix. A sparse matrix is a matrix where most of the values are 0. 

from sklearn.feature_extraction.text import CountVectorizer


# In[33]:

#In this case we will just specify the analyzer argument in order to set it to be our own previously defined function
#which is text_process

#We define an object called bag_of_words_transformer
bag_of_words_transformer = CountVectorizer(analyzer=text_process)


# In[34]:

#Next fit the bag_of_words model to that column called message in the messages dataframe
bag_of_words_transformer.fit(messages['message'])
print len(bag_of_words_transformer.vocabulary_)
#The warning below is because of some weird unicode in the text message such as the pound symbol 


# In[41]:

message4 = messages['message'][3]
print message4


# In[42]:

#See if it works!
bow4 = bag_of_words_transformer.transform([message4])
print bow4
print bow4.shape
#Numbers like 2924 and 3293 - stand for word number 2924 - word number 3293
#It looks like message 4 has 1 unique word (after removing the common stop words) 


# In[43]:

bow4.dtype


# In[44]:

print bag_of_words_transformer.get_feature_names()[2924]


# In[45]:

#Now we can use dot transform on our Bag-of-Words object 
#and transform the entire DataFrame of messages. Let's go ahead and check out how the bag-of-words counts 
#for the entire collection of text messages is a large, sparse matrix:
messages_bow = bag_of_words_transformer.transform(messages['message'])


# In[46]:

print 'Shape of Sparse Matrix: ', messages_bow.shape
print 'Amount of Non-Zero occurences: ', messages_bow.nnz
#nnz is part of scipy.sparse.dia_matrix.nnz
# outputs the number of nonzero values
#explicit zero values are included in this number
print 'sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))


# In[47]:

#TF-IDF stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used 
#in information retrieval and text mining. This weight is a statistical measure used to evaluate 
#how important a word is to a document in a collection or corpus. The importance increases 
#proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus.
#If we are trying to compare a text message of 100 characters and you have a few others that are text messages of 900 characters
#You will need to divide the number of times term t appears in a document by the Total number of terms in the document
#otherwise, you'll begin to skew your weights for common terms 
#To expand that  idea, we have:
#IDF: Inverse Document Frequency, which measures how important a term is. 
#While computing TF, all terms are considered equally important. 
#However it is known that certain terms, such as "is", "of", and "that", 
#may appear a lot of times but have little importance. Thus we need to weigh down the frequent 
#terms while scale up the rare ones, by computing the following:
#IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
#Example:

#Consider a document containing 100 words wherein the word cat appears 3 times.

#The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03. 
#Now, assume we have 10 million documents and the word cat appears 
#in one thousand of these. Then, the inverse document frequency (i.e., idf) 
#is calculated as log(10,000,000 / 1,000) = 4. 
#Thus, the Tf-idf weight is the product of these quantities: 0.03 * 4 = 0.12.


# In[48]:

#Import from scikit-learn's feature extraction library, import the TfidTransformer
from sklearn.feature_extraction.text import TfidfTransformer


# In[49]:

#Create an object called tfid_transformer and set it equal to TfidTransformer for which 
#you need to call the fit method and pass the fit mehod on to that bag of words object called messages_bow
tfidf_transformer = TfidfTransformer().fit(messages_bow)


# In[50]:

#We next, need to transform the bag-of-words counts as a vector
#Recall that messages_bow is a sparse matrix - this sparse matrix was generated by the bag of words transformer
##This is essentially a 2 dimensional matrix
#The entire sparse matrix needs to transformed by the 
#tfidf_transformer - into a series containing all the unique words (in the form of numbers for each unique word)
#and the number of times these unique words appear in teach message in the data frame
#To transform the entire bag-of-words corpus into TF-IDF corpus at once:
messages_tfidf = tfidf_transformer.transform(messages_bow)
print messages_tfidf.shape


# In[51]:

#Take a look at how the tfidf transformer words for a single message
#Recall that we applied the bag of words transformer to a single message
#and set this equal to bow4
#Let's apply the TFIDF transformer to bow4
tfidf4 = tfidf_transformer.transform(bow4)
print tfidf4


# In[52]:

#Messages are represented as vectors - we can start training a multi-class classifier
#Import multinomial naive bayes
from sklearn.naive_bayes import MultinomialNB


# In[93]:

#Define a variable called multi_class_model
multi_class_model = MultinomialNB(alpha=0.044701285593).fit(messages_tfidf, messages['label'])


# In[91]:

#Try our model on a single message
print 'predicted: ', multi_class_model.predict(tfidf4)
#Next, print the label of the 100th message (remember python indexes from 0)
print 'expected: ', messages.label[100]


# In[86]:

# Randomized Search for Algorithm Tuning (goal is to find best learning rate - alpha)
#Experimented with cv values from 5 to 100 to get the best alpha value in terms of the best possible accuracy score
#The grid search algorithm proved to be more computionally expensive
import numpy as np
from scipy.stats import uniform as sp_rand
from sklearn import datasets
from sklearn.grid_search import RandomizedSearchCV
# load the dataset
dataset = messages
# prepare a uniform distribution to sample for the alpha parameter
param_grid = {'alpha': sp_rand()}
# create and fit a Multinomial Naive Bayes model, testing random alpha values
model = MultinomialNB()
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, cv=30)
rsearch.fit(messages_tfidf, messages['label'])
print(rsearch)
# summarize the results of the random parameter search
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)


# In[84]:

from sklearn import cross_validation


# In[94]:

#If you evaluate your algorithm on the same dataset that you train it on, 
#it's impossible to know if it's performing well because it overfit itself to the noise, 
#or if it actually is a good algorithm.

#Luckily, cross validation is a simple way to avoid overfitting. 
#To cross validate, you split your data into some number of parts (or "folds"). 
#Lets use 3 as an example. You then do this:

#Combine the first two parts, train a model, make predictions on the third.

#Combine the first and third parts, train a model, make predictions on the second.

#Combine the second and third parts, train a model, make predictions on the first.

#This way, we generate predictions for the whole dataset without ever evaluating accuracy on 
#the same data we train our model using.
scores = cross_validation.cross_val_score(multi_class_model, messages_tfidf, messages['label'], cv=30)


# In[95]:

print(scores.mean())


# In[88]:

#Now we want to determine how well our model will do overall on the entire dataset (the messages dataframe). 
#Let's beginby getting all the predictions:
all_predictions = multi_class_model.predict(messages_tfidf)
print all_predictions


# In[89]:

#We can use SciKit Learn's built-in classification report, which returns precision, recall, 
#f1-score, and a column for support (meaning how many cases supported that classification).
#From the metrics library of scikit-learn, import classification_report
#Print the classification report for the messages column in the message data frame, based on the 
#serifrom sklearn.metrics import classification_report
from sklearn.metrics import classification_report
print classification_report(messages['label'], all_predictions)


# In[ ]:



