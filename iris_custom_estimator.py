import pandas as pd
import tensorflow as tf
from tensorflow.python.estimator.canned.linear import _compute_fraction_of_zero
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.training import training_util
from tensorflow.python.training.ftrl import FtrlOptimizer


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.shuffle(1000).repeat().batch(batch_size)


none = None


def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not none, "batch_size must not be None"
    return dataset.batch(batch_size)


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
LEARNING_RATE = 0.3

weight_column = None
label_vocabulary = None
loss_reduction = tf.losses.Reduction.SUM

head = tf.contrib.estimator.multi_class_head(3, weight_column=weight_column,
                                             label_vocabulary=label_vocabulary,
                                             loss_reduction=loss_reduction)


def train_op_fn(loss):
    opt = FtrlOptimizer(learning_rate=LEARNING_RATE)
    return opt.minimize(loss, global_step=training_util.get_global_step())


def model_fn(features, labels, mode, config):
    def logit_fn(features):
        cols_to_vars = {}
        return feature_column_lib.linear_model(
            features=features,
            feature_columns=feature_columns,
            units=head.logits_dimension,
            cols_to_vars=cols_to_vars)

    logits = logit_fn(features=features)

    return head.create_estimator_spec(
        features=features,
        mode=mode,
        logits=logits,
        labels=labels,
        train_op_fn=train_op_fn)


classifier = tf.estimator.Estimator(model_fn=model_fn)

classifier.train(
    input_fn=lambda: train_input_fn(train_x, train_y, 100),
    steps=2000)

eval_result = classifier.evaluate(
    input_fn=lambda: eval_input_fn(test_x, test_y, 100))

print(eval_result)
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
