
# coding: utf-8

# In[1]:

import nltk
nltk.download()


# In[2]:

from nltk.corpus import names


# In[3]:

pwd


# In[57]:

list_of_names_labels = ([(name, 'male') for name in names.words('male.txt')] + 
                       [(name, 'female') for name in names.words('female.txt')])


# In[54]:

def last_letter(word):
    return {'last_letter': word[-1]}


# In[56]:

#Test out function
last_letter('Preetha')


# In[58]:

import random


# In[59]:

random.shuffle(list_of_names_labels)


# In[62]:

feature_set = [(last_letter(name), gender) for (name, gender) in list_of_names_labels]
feature_set


# In[63]:

train_set, test_set = feature_set[2400:], featuresets[:5600]
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[65]:

#Trying out classifier
classifier.classify(last_letter('Preetha'))


# In[66]:

classifier.classify(last_letter('Priya'))


# In[67]:

classifier.classify(last_letter('Raju'))


# In[68]:

classifier.classify(last_letter('Mary'))


# In[69]:

classifier.classify(last_letter('Rajan'))


# In[46]:

classifier.classify(last_letter('Anjana'))


# In[70]:

classifier.classify(last_letter('Jack'))


# In[71]:

classifier.classify(last_letter('Leo'))


# In[43]:

print(nltk.classify.accuracy(classifier, test_set))


# In[47]:

print(nltk.classify.accuracy(classifier, train_set))


# In[ ]:



