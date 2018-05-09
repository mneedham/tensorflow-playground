import csv

import pandas as pd
import tensorflow as tf

from matplotlib import pyplot as plt

from neo4j.v1 import GraphDatabase, basic_auth

driver = GraphDatabase.driver("bolt://localhost", auth=basic_auth("neo4j", "neo"))

genres_query = """\
MATCH (genre:Genre)
WITH genre ORDER BY genre.name
WITH collect(id(genre)) AS genres
MATCH (m:Movie)-[:IN_GENRE]->(genre)
WITH genres, id(m) AS source, collect(id(genre)) AS target
RETURN source, [g in genres | CASE WHEN g in target THEN 1 ELSE 0 END] AS genres
"""

movies_query = """\
MATCH (m:Movie)
RETURN collect(id(m)) AS movies
"""

with driver.session() as session:
    result = session.run(genres_query)
    all_genres = pd.DataFrame([dict(row) for row in result])
    movies = [row["movies"] for row in session.run(movies_query)][0]

all_movies = []
with open("movies.emb", "r") as movies_file:
    next(movies_file)
    reader = csv.reader(movies_file, delimiter=" ")
    for row in reader:
        movie_id = row[0]
        if int(movie_id) in movies:
            all_movies.append({"source": int(movie_id), "embedding": [float(item) for item in row[1:]]})

all_movies = pd.DataFrame(all_movies)

everything = pd.merge(all_movies, all_genres, on='source')

train_index = int(len(everything) * 0.9)
train_data = everything[:train_index]
test_data = everything[train_index:]


# separate train data
train_x = train_data.ix[:, 'embedding']
train_y = train_data.ix[:, 'genres']

# separate test data
test_x = test_data.ix[:, 'embedding']
test_y = test_data.ix[:, 'genres']

# placeholders for inputs and outputs
X = tf.placeholder(tf.float32, [None, 100])
Y = tf.placeholder(tf.float32, [None, 20])

# weight and bias
weight = tf.Variable(tf.zeros([100, 20]))
bias = tf.Variable(tf.zeros([20]))

# output after going activation function
# output = tf.nn.softmax(tf.matmul(X, weight) + bias)
logits = tf.matmul(X, weight)

output = tf.nn.sigmoid(logits + bias)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=Y)

# cost funciton
cost = tf.reduce_mean(tf.square(Y - output))
# train model
train = tf.train.AdamOptimizer(0.01).minimize(cost)

# check sucess and failures
# success = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
success = tf.equal(tf.round(output), Y)

# calculate accuracy
accuracy = tf.reduce_mean(tf.cast(success, tf.float32)) * 100

# initialize variables
init = tf.global_variables_initializer()

# start the tensorflow session
with tf.Session() as sess:
    costs = []
    sess.run(init)
    # train model 1000 times
    for i in range(200):
        _, c = sess.run([train, cost], {X: [t for t in train_x.as_matrix()], Y: [t for t in train_y.as_matrix()]})
        costs.append(c)
        if i % 50 == 0:
            print("Iteration: {0}".format(i))
            print("Accuracy: %.2f" % accuracy.eval({X: [t for t in test_x.as_matrix()], Y: [t for t in test_y.as_matrix()]}))


    print("Training finished!")
    print("Accuracy: %.2f" % accuracy.eval({X: [t for t in test_x.as_matrix()], Y: [t for t in test_y.as_matrix()]}))
    plt.plot(range(200), ctenssts)
    plt.title("Cost Variation")
    plt.show()
