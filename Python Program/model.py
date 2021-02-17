# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tensorflow as tf
import random

# import our chat-bot contexts file
import json
with open('contexts.json') as json_data:
    contexts = json.load(json_data)

words = []
classes = []
documents = []
ignore_words = ['?']
#nltk.download('punkt')

# loop through each sentence in our contexts patterns
for intent in contexts['contexts']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        print(w)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])
#print (train_x, train_y)

D = len(train_x[0])
M1 = 8
M2 = 8
K = len(train_y[0])

# tensor flow variables are not the same as regular Python variables
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.02))


def forward(X, W1, b1, W2, b2, W3, b3):
    Z1 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    Z2 = tf.nn.sigmoid(tf.matmul(Z1, W2) + b2)
    return tf.matmul(Z2, W3) + b3


tfX = tf.placeholder(tf.float32, [None, D], name="input")
tfY = tf.placeholder(tf.float32, [None, K])

W1 = init_weights([D, M1]) # create symbolic variables
b1 = init_weights([M1])
W2 = init_weights([M1, M2])
b2 = init_weights([M2])
W3 = init_weights([M2, K])
b3 = init_weights([K])

logits = forward(tfX, W1, b1, W2, b2, W3, b3)

cost = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(
    labels=tfY,
    logits=logits
  )
)
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost) # Building an optimizer


predict_op = tf.nn.sigmoid(logits, name="output")

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(150000):
    sess.run(train_op, feed_dict={tfX: train_x, tfY: train_y})
    pred = sess.run(predict_op, feed_dict={tfX: train_x})
    if i % 100 == 0:                               #This is just for testing purpose don't worry about it
        print(float(i)/150000)
        #print("The predicted value is ")
        #print(pred)
        #print("label is ")
        #print(Y)
        #print("Accuracy:", np.mean(Y == pred))


saver = tf.train.Saver()
save_path = saver.save(sess, "saved_chatbot.ckpt")
print("Model saved to %s" % save_path)


# save all of our data structures
import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )
