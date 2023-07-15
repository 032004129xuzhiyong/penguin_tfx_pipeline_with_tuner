import keras_tuner
import tensorflow as tf
import tensorflow_transform as tft
from tfx import v1 as tfx
from typing import List
from tensorflow import keras
from tfx_bsl.public import tfxio
from absl import logging

from models import features
from models import constants

from tensorflow_metadata.proto.v0 import schema_pb2

class MyHyperModel(keras_tuner.HyperModel):
  def __init__(self, feature_list, **kwargs):
    self.feature_list = feature_list
    super().__init__(**kwargs)
  def build(self,hp :keras_tuner.HyperParameters):
    return _build_keras_model(self.feature_list,
                       hp.Int('num_dense_layer',constants.MIN_NUM_LAYERS,constants.MAX_NUM_LAYERS),
                       hp.Choice('dense_activation',constants.ACTIVATION_LIST))
  def fit(self, hp, model, **kwargs):
    return model.fit(**kwargs)

def _build_keras_model(feature_list: List[str],num_dense_layer,dense_activation) -> tf.keras.Model:
  """Creates a DNN Keras model for classifying penguin data.

  Args:
    feature_list: List of feature names.

  Returns:
    A Keras Model.
  """
  # The model below is built with Functional API, please refer to
  # https://www.tensorflow.org/guide/keras/overview for all API options.
  inputs = [keras.layers.Input(shape=(1,), name=f) for f in feature_list]
  d = keras.layers.concatenate(inputs)
  for _ in range(num_dense_layer):
    d = keras.layers.Dense(constants.HIDDEN_LAYER_UNITS, activation=dense_activation)(d)
  outputs = keras.layers.Dense(
      constants.OUTPUT_LAYER_UNITS, activation='softmax')(
          d)

  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
      optimizer=keras.optimizers.Adam(constants.LEARNING_RATE),
      loss='sparse_categorical_crossentropy',
      metrics=[keras.metrics.SparseCategoricalAccuracy()])

  model.summary(print_fn=logging.info)
  return model


def _input_fn(file_pattern: List[str],
              data_accessor: tfx.components.DataAccessor,
              schema: schema_pb2.Schema,
              label: str,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    schema: A schema proto of input data.
    label: Name of the label.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(batch_size=batch_size, label_key=label),
      schema).repeat()



def tuner_fn(fn_args: tfx.components.FnArgs) -> tfx.components.TunerFnResult:
  if fn_args.transform_graph_path is None:  # Transform is not used.
    tf_transform_output = None
    schema = tfx.utils.parse_pbtxt_file(fn_args.schema_path,
                                        schema_pb2.Schema())
    feature_list = features.FEATURE_KEYS
    label_key = features.LABEL_KEY
  else:
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    schema = tf_transform_output.transformed_metadata.schema
    feature_list = [features.transformed_name(f) for f in features.FEATURE_KEYS]
    label_key = features.transformed_name(features.LABEL_KEY)

  mirrored_strategy = tf.distribute.MirroredStrategy()
  train_batch_size = (
      constants.TRAIN_BATCH_SIZE * mirrored_strategy.num_replicas_in_sync)
  eval_batch_size = (
      constants.EVAL_BATCH_SIZE * mirrored_strategy.num_replicas_in_sync)

  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      schema,
      label_key,
      batch_size=train_batch_size)
  eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      schema,
      label_key,
      batch_size=eval_batch_size)

  with mirrored_strategy.scope():
    tuner = keras_tuner.BayesianOptimization(MyHyperModel(feature_list),
                                             objective=keras_tuner.Objective('val_sparse_categorical_accuracy','max'),
                                             max_trials=10
                                             )
  return tfx.components.TunerFnResult(tuner=tuner,fit_kwargs={'x':train_dataset,
                                                              'validation_data':eval_dataset,
                                                              'epochs':2,
                                                              'steps_per_epoch':fn_args.train_steps,
                                                              'validation_steps':fn_args.eval_steps
                                                              })