import nltk     #a suite of libraries and programs for symbolic and statistical natural language processing for English written in the Python programming language.

from nltk.stem import WordNetLemmatizer

nltk.download('popular',quiet=True) #for downloading popular packages when it is executed for 1st time(punkt and wordnet).

nltk.download('punkt')    #punkt is a sentence tokenizer. This tokenizer divides a text into a list of sentences, by using an unsupervised algorithm
                          #to build a model for abbreviation words,collocations, and words that start sentences.

nltk.download('wordnet')        #Wordnet is an large, freely and publicly available lexical database for the English language aiming to establish structured semantic relationships
                                #between words. It offers lemmatization capabilities as well and is one of the earliest and most commonly used lemmatizers.

lemmatizer = WordNetLemmatizer()
import json


intents_file = open('intents.json').read()
intents = json.loads(intents_file)    #loading the intents.json file to intents.


words=[]
classes = []
documents = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #for tokenize (Tokenization is the act of breaking up a sequence of strings into pieces such as words, keywords, phrases, symbols and other elements called tokens.)
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        documents.append((word, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
print(words)


#Lemmatization is the process of converting a word to its base form. The difference between stemming and lemmatization is,
#lemmatization considers the context and converts the word to its meaningful base form, whereas stemming just removes
#the last few characters, often leading to incorrect meanings and spelling errors.


words = [lemmatizer.lemmatize(w.lower()) for w in words]    # lemmaztize and lower each word and remove duplicates
words = (list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique lemmatized words", words)

  #-----------------------------------------------------------------------------------------------------------------------------------------------------------------#
import pickle

#Pickle in Python is primarily used in serializing and deserializing a Python object structure.
#In other words, it’s the process of converting a Python object into a byte stream to store it in a
#file/database, maintain program state across sessions, or transport data over the network.

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))               #wb means write binary


training = []      # create our training data

output_empty = [0] * len(classes)      # create an empty array for our output

for doc in documents:                   # training set, bag of words for each sentence
    
    bag = []                            # initialize our bag of words
    
    pattern_words = doc[0]              # list of tokenized words for the pattern
    
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]      # lemmatize each word - create base word, in attempt to represent related words
   
    for word in words:                       # create our bag of words array with 1, if word match found in current pattern
        bag.append(1) if word in pattern_words else bag.append(0)
        
     # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])       #appended to training data
    

import random

#to generate random numbers in Python by using random module. Python offers random module that can generate random numbers.
#These are pseudo-random number as the sequence of number generated depends on the seed. If the seeding value is same,
#the sequence will be the same.


import numpy as np

#NumPy is an open-source numerical Python library. NumPy contains a multi-dimensional array and matrix data structures.
#It can be utilised to perform a number of mathematical operations on arrays such as trigonometric, statistical, and algebraic routines.

random.shuffle(training)             # shuffle our features and turn into np.array
training = np.array(training)     
train_x = list(training[:,0])        #create train and test lists. X - patterns, Y - intents
train_y = list(training[:,1])
print("Training data created")
        
  #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
                                                                        #Implementing Neural Network



from keras.models import Sequential

#Keras is a powerful and easy-to-use free open source Python library for developing and evaluating deep learning models.
#It wraps the efficient numerical computation libraries Theano and TensorFlow and allows you to define and train neural network models in just a few lines of code.
  #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#The sequential API allows you to create models layer-by-layer for most problems.
#It is limited in that it does not allow you to create models that share layers or have multiple inputs or outputs.

from keras.layers import Dense, Activation, Dropout

#A Dense layer feeds all outputs from the previous layer to all its neurons, each neuron providing one output to the next layer.
#It's the most basic layer in neural networks. A Dense(10) has ten neurons.
  #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#Dropout is a technique used to prevent a model from overfitting.
#Dropout works by randomly setting the outgoing edges of hidden units(neurons that make up hidden layers) to 0 at each update of the training phase.
  #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#The activation function is a mathematical “gate” in between the input feeding the current neuron and its output going to the next layer.
#It can be as simple as a step function that turns the neuron output on and off, depending on a rule or threshold.
  #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

from keras.optimizers import SGD

#Stochastic gradient descent (SGD) is an iterative method for optimizing an objective function
#with suitable smoothness properties (e.g. differentiable or subdifferentiable).
   #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#SGD is the simplest algorithm both conceptually and in terms of its behavior.
#Given a small enough learning rate, SGD always simply follows the gradient on the cost surface.
#The new weights generated on each iteration will always be strictly better than old ones from the previous iteration.
   #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#The rectified linear activation function or ReLU is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero.
#It has become the default activation function for many types of neural networks because a model that uses it is easier to train and often achieves better performance.
   #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#Softmax is often used as the activation for the last layer of a classification network because the result could be interpreted as a probability distribution.
#The softmax of each vector x is computed as exp(x) / tf. reduce_sum(exp(x)) . The input values in are the log-odds of the resulting probability.

model = Sequential()    
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))        #input layer contains 128 neurons   #Rectified Linear Unit (ReLU)
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))                                         #hidden layer contains 64 neurons
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))                         #output layer contains no of neurons equal to no of intents to predict output intent


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)       	   # Compile model. Stochastic gradient descent with Nesterov accelerated gradient 
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


fitting = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)          #fitting and saving the model 
model.save('chatbot_model.h5', fitting)

print("model created")





































