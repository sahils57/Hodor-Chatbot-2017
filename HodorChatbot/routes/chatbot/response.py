# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# restore all of our data structures
import pickle
data = pickle.load( open( os.getcwd()+"/../routes/chatbot/training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# things we need for Tensorflow
import numpy as np
import tensorflow as tf
import random

# import our chat-bot contexts file
import json
with open(os.getcwd()+"/../routes/chatbot/contexts.json") as json_data:
    contexts = json.load(json_data)

sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph(os.getcwd()+"/../routes/chatbot/saved_chatbot.ckpt.meta")
new_saver.restore(sess, os.getcwd()+"/../routes/chatbot/saved_chatbot.ckpt")

data_in = sess.graph.get_tensor_by_name("input:0")
predict_op = sess.graph.get_tensor_by_name("output:0")


#print('checkpoint')
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))


ERROR_THRESHOLD = 0.25
def classify(sentence):
    # generate probabilities from the model
    #results = model.predict([bow(sentence, words)])[0]
    results = sess.run(predict_op, feed_dict= {data_in:[bow(sentence, words)]})[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    if show_details:
        print(results)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            #print('\nresults: \n', results)
            for i in contexts['contexts']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # a random response from the intent
                    print(random.choice(i['responses']))
                    return 1

            results.pop(0)

response(sys.argv[1])
