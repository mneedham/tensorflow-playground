import pandas as pd
import tensorflow as tf


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.shuffle(1000).repeat().batch(batch_size)


def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "batch_size must not be None"
    return dataset.batch(batch_size)


def predict_input_fn(features, batch_size):
    features = dict(features)
    dataset = tf.data.Dataset.from_tensor_slices(features)
    assert batch_size is not None, "batch_size must not be None"
    return dataset.batch(batch_size)


SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# read data from csv
train_data = pd.read_csv("iris_training.csv", names=['f1', 'f2', 'f3', 'f4', 'f5'])
test_data = pd.read_csv("iris_test.csv", names=['f1', 'f2', 'f3', 'f4', 'f5'])

# separate train data
train_x = train_data[['f1', 'f2', 'f3', 'f4']]
train_y = train_data.ix[:, 'f5']

# separate test data
test_x = test_data[['f1', 'f2', 'f3', 'f4']]
test_y = test_data.ix[:, 'f5']

# Define feature columns
feature_columns = [tf.feature_column.numeric_column(key=key) for key in train_x.keys()]

# Define classifier
classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=3)

classifier.train(
    input_fn=lambda: train_input_fn(train_x, train_y, 100),
    steps=2000)

eval_result = classifier.evaluate(
    input_fn=lambda: eval_input_fn(test_x, test_y, 100))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'f1': [5.1, 5.9, 6.9],
    'f2': [3.3, 3.0, 3.1],
    'f3': [1.7, 4.2, 5.4],
    'f4': [0.5, 1.5, 2.1],
}

predictions = classifier.predict(
    input_fn=lambda: predict_input_fn(predict_x, batch_size=100))

template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print(template.format(SPECIES[class_id], 100 * probability, expec))
