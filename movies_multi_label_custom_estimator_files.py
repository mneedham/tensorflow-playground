import pandas as pd
import numpy as np
import csv
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.training import training_util
from tensorflow.python.training.ftrl import FtrlOptimizer

from neo4j.v1 import GraphDatabase, basic_auth

none = None


def train_input_fn(features, labels, batch_size):
    labels = tf.constant(np.array([np.array(item)
                                   for item in labels.values]))
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.shuffle(1000).repeat().batch(batch_size)


def eval_input_fn(features, labels, batch_size):
    labels = tf.constant(np.array([np.array(item)
                                   for item in labels.values]))

    features = dict(features)
    inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    assert batch_size is not none, "batch_size must not be None"
    return dataset.batch(batch_size)


def model_fn(features, labels, mode, config):
    def train_op_fn(loss):
        print(loss)
        opt = FtrlOptimizer(learning_rate=LEARNING_RATE)
        return opt.minimize(loss,
                            global_step=training_util.get_global_step())

    def logit_fn(features):
        return feature_column_lib.linear_model(
            features=features,
            feature_columns=feature_columns,
            units=head.logits_dimension)

    return head.create_estimator_spec(
        features=features,
        mode=mode,
        logits=logit_fn(features=features),
        labels=labels,
        train_op_fn=train_op_fn)


# host = "bolt://localhost"
# password = "neo"

host = "bolt://54.197.87.95:32952"
password = "pennant-automation-pacific"

driver = GraphDatabase.driver(host, auth=basic_auth("neo4j", password))

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
            all_movies.append({"source": int(movie_id),
                               "embedding": [float(item)
                                             for item in row[1:]]})

all_movies = pd.DataFrame(all_movies)

everything = pd.merge(all_movies, all_genres, on='source')

train_index = int(len(everything) * 0.9)
train_data = everything[:train_index]
test_data = everything[train_index:]

train_x = train_data.ix[:, "embedding"]
train_x = pd.DataFrame(np.array([np.array(item) for item in train_x.values]))
train_x.columns = [str(col) for col in train_x.columns.get_values()]

train_y = train_data.ix[:, 'genres']

# separate test data
test_x = test_data.ix[:, "embedding"]
test_x = pd.DataFrame(np.array([np.array(item) for item in test_x.values]))
test_x.columns = [str(col) for col in train_x.columns.get_values()]

test_y = test_data.ix[:, 'genres']

# Define feature columns
feature_columns = [tf.feature_column.numeric_column(key=key)
                   for key in train_x.keys()]

# Define classifier
LEARNING_RATE = 0.3
loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE

head = tf.contrib.estimator.multi_label_head(20, weight_column=none,
                                             label_vocabulary=none,
                                             loss_reduction=loss_reduction)

classifier = tf.estimator.Estimator(model_fn=model_fn)

classifier.train(
    input_fn=lambda: train_input_fn(train_x, train_y, 100),
    steps=2000)

eval_result = classifier.evaluate(
    input_fn=lambda: eval_input_fn(test_x, test_y, 100))

print(eval_result)
print('\nTest set AUC: {auc:0.3f}\n'.format(**eval_result))

temp = classifier.predict(input_fn=lambda: eval_input_fn(train_x, train_y, 100))
for i in range(10):
    print((next(temp)['logits']))
    print((next(temp)['probabilities']))