# Copyright 2023 Google LLC.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.mit.edu/~amini/LICENSE.md
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer-based tabular data learning architecture.

Copyright (c) 2019 DreamQuark

TabNet:
https://arxiv.org/abs/1908.07442

The implementation is a refactored code from:
https://github.com/dreamquark-ai/tabnet
"""
import functools
import math
from typing import Any

from flax.core import frozen_dict
import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict

from optimus.agent_lib import base_agent

FrozenDict = frozen_dict.FrozenDict

DEFAULT_HYPERPARAMETERS = config_dict.ConfigDict(
    dict(
        attention_gamma=1.3,
        prediction_layer_dimension=8,
        attention_layer_dimension=8,
        successive_network_steps=3,
        categorical_embedding_dimensions=[3],
        independent_glu_layers=2,
        shared_glu_layers=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.98,
        mask_type="sparsemax",
        shared_decoder_layers=1,
        independent_decoder_layers=1,
        model_data_type="float32",
        agent_name="tabnet",
        replace_nans_in_prediction=False,
    )
)


class EmbeddingGenerator(nn.Module):
  """Generates embeddings in the model.

  Attributes:
    input_dimensions: The number of feature columns in the input table.
    categorical_dimensions: The number of dimensions in each of the categorical
      columns of the input table.
    categorical_indexes: Indices of categorical columns in the input table.
    categorical_embedding_dimensions: The size of the embedding of categorical
      features if int, all categorical features will have same embedding size if
      list of int, every corresponding feature will have specific size.
    model_data_type: Data type to use within the model.
    embedding_initializer: Embeddings initializer.
    post_embedding_dimension: The post embedding dimension.
  """

  input_dimensions: int
  categorical_dimensions: tuple[int, ...] | None = None
  categorical_indexes: tuple[int, ...] | None = None
  categorical_embedding_dimensions: tuple[int, ...] | None = None
  model_data_type: jnp.dtype = jnp.float32
  embedding_initializer: Any = nn.initializers.normal(stddev=1**0.5)

  @functools.cached_property
  def _sorted_categorical_indexes(self) -> tuple[int, ...]:
    """Returns the sorted_indexes."""
    if self.categorical_indexes:
      return tuple([
          self.categorical_indexes.index(i)
          for i in sorted(self.categorical_indexes)
      ])
    else:
      raise ValueError("There is no categorical_indexes to sort.")

  @functools.cached_property
  def _verified_categorical_embedding_dimensions(
      self,
  ) -> tuple[int, ...] | None:
    """Returns the verified_categorical_embedding_dimensions."""
    if len(self.categorical_embedding_dimensions) == 1:
      categorical_embedding_dimensions = (
          self.categorical_embedding_dimensions * len(self.categorical_indexes)
      )
    else:
      categorical_embedding_dimensions = self.categorical_embedding_dimensions
    if len(categorical_embedding_dimensions) != len(
        self.categorical_dimensions
    ):
      raise ValueError(
          "categorical_embedding_dimensions and categorical_dimensions must"
          " be lists of same length, got"
          f" {len(self.categorical_embedding_dimensions)} and"
          f" {len(self.categorical_dimensions)}"
      )
    return categorical_embedding_dimensions

  @functools.cached_property
  def post_embedding_dimension(self) -> int:
    """Returns the post embedding dimension."""
    if not self.categorical_dimensions and not self.categorical_indexes:
      return self.input_dimensions
    return (
        self.input_dimensions
        + sum(self._verified_categorical_embedding_dimensions)
        - len(self._verified_categorical_embedding_dimensions)
    )

  def setup(self) -> None:
    """Sets up the module definition when it's called.

    Raises:
      ValueError: An error when length of categorical_dimensions is not equal to
      categorical_indexes. Or when length of categorical_embedding_dimensions
      is not equal to categorical_dimensions.
    """
    if not self.categorical_dimensions and not self.categorical_indexes:
      self.skip_embedding = True
      return
    if not self.categorical_dimensions or not self.categorical_indexes:
      raise ValueError(
          "Both categorical_dimensions and categorical_indexes "
          "are required if there are categorical columns."
      )
    if len(self.categorical_dimensions) != len(self.categorical_indexes):
      raise ValueError(
          "The lists categorical_dimensions and categorical_indexes must have"
          " the same length."
      )
    self.skip_embedding = False
    categorical_dimensions = [
        self.categorical_dimensions[i] for i in self._sorted_categorical_indexes
    ]
    embedding = functools.partial(
        nn.Embed,
        embedding_init=self.embedding_initializer,
        dtype=self.model_data_type,
    )
    processed_categorical_embedding_dimensions = [
        self._verified_categorical_embedding_dimensions[i]
        for i in self._sorted_categorical_indexes
    ]
    self.embeddings = [
        embedding(
            num_embeddings=categorical_dimensions + 1,
            features=embedding_dimensions,
        )
        for categorical_dimensions, embedding_dimensions in zip(
            categorical_dimensions,
            processed_categorical_embedding_dimensions,
        )
    ]
    self.continuous_indexes = [
        0
        if i in self.categorical_indexes
        else 1
        for i in range(self.input_dimensions)
    ]

  @nn.compact
  def __call__(self, *, input_x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass implementation for the layer.

    Args:
      input_x: An input array used in the embedding process.

    Returns:
      The output of the forward pass.

    Raises:
      ValueError: An error when cat_feat_counter is higher than the nuber of
        embedding layers.
    """
    if self.skip_embedding:
      return input_x
    columns = []
    categorical_feature_counter = 0
    for feature_initial_index, is_continuous in enumerate(
        self.continuous_indexes
    ):
      if is_continuous:
        output = jnp.array(
            input_x[:, feature_initial_index], self.model_data_type
        ).reshape(-1, 1)
        columns.append(output)
      else:
        if categorical_feature_counter > len(self.embeddings):
          raise ValueError(
              "There're more categorical features than embedding layers."
          )
        embedding_layer = self.embeddings[categorical_feature_counter]
        embedding_input = jnp.array(
            input_x[:, feature_initial_index], dtype=jnp.int32
        )
        embedding_output = embedding_layer(inputs=embedding_input)
        columns.append(embedding_output)
        categorical_feature_counter += 1
    return jnp.concatenate(columns, axis=1)


class RandomObfuscator(nn.Module):
  """Used to randomly obfuscate column information during pre-training.

  Attributes:
    pretraining_ratio: The ratio of data to randomly obfuscate.
  """

  pretraining_ratio: float = 0.2

  @nn.compact
  def __call__(
      self, *, input_x: jnp.ndarray
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Forward pass implementation for the layer.

    Args:
      input_x: An input array where random information will be obfuscated.

    Returns:
      The output of the forward pass.
    """
    bernoulli_rng = self.make_rng("bernoulli_rng")
    obfuscated_vars = jax.random.bernoulli(
        bernoulli_rng, self.pretraining_ratio * jnp.ones(input_x.shape)
    )
    masked_input = jnp.multiply(1 - obfuscated_vars, input_x)
    return masked_input, obfuscated_vars


class GhostBatchNormalization(nn.Module):
  """Implementation of Ghost Batch Normalization.

  It enables significant decrease in the generalization gap without
  increasing the number of updates. For more details see:
  https://arxiv.org/abs/1705.08741

  Attributes:
    virtual_batch_size: Batch size for Ghost Batch Normalization.
    momentum: Momentum value for the BatchNorm layer.
    model_data_type: Data type to use within the model.
  """

  virtual_batch_size: int = 128
  momentum: float = 0.99
  model_data_type: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self, *, input_x: jnp.ndarray, train: bool = False
  ) -> jnp.ndarray:
    """Forward pass implementation for the layer.

    Args:
      input_x: An input array to be processed by the Global Batch Normalization
        layer.
      train: An indicator whether the pass is during training or not.

    Returns:
      The output of the forward pass.
    """
    batch_norm = nn.BatchNorm(
        momentum=self.momentum,
        use_running_average=not train,
        dtype=self.model_data_type,
    )
    indices_or_sections = math.ceil(input_x.shape[0] / self.virtual_batch_size)
    chunks = jnp.split(input_x, indices_or_sections, 0)
    result = [batch_norm(x=x) for x in chunks]
    return jnp.concatenate(result, axis=0)


class GatedLinearUnitLayer(nn.Module):
  """Implementation of a Gated Linear Unit Layer.

  See:
  https://arxiv.org/abs/2002.05202

  Attributes:
    input_dimensions: The number of feature columns in the input table.
    output_dimensions: The number of possible actions.
    fully_connected_layer: A fully-connected Dense layer.
    virtual_batch_size: Batch size for Ghost Batch Normalization.
    momentum: Momentum value for the BatchNorm layer.
    model_data_type: Data type to use within the model.
  """

  input_dimensions: int
  output_dimensions: int
  fully_connected_layer: nn.Module | None = None
  virtual_batch_size: int = 128
  momentum: float = 0.98
  model_data_type: jnp.dtype = jnp.float32

  def setup(self) -> None:
    """Sets up the module definition when it's called."""
    gain_value = jnp.sqrt(
        (self.input_dimensions + 2 * self.output_dimensions)
        / jnp.sqrt(self.input_dimensions)
    )
    kernel_init = nn.initializers.variance_scaling(
        scale=gain_value, mode="fan_avg", distribution="truncated_normal"
    )
    if self.fully_connected_layer:
      self.fc_layer = self.fully_connected_layer
    else:
      self.fc_layer = nn.Dense(
          2 * self.output_dimensions,
          use_bias=False,
          kernel_init=kernel_init,
          dtype=self.model_data_type,
      )
    self.ghost_batch_norm = GhostBatchNormalization(
        virtual_batch_size=self.virtual_batch_size,
        momentum=self.momentum,
        model_data_type=self.model_data_type,
    )

  @nn.compact
  def __call__(
      self, *, input_x: jnp.ndarray, train: bool = False
  ) -> jnp.ndarray:
    """Forward pass implementation for the layer.

    Args:
      input_x: An input array to be precessed by GLU layer.
      train: An indicator whether the pass is during training or not.

    Returns:
      The output of the forward pass.
    """
    fully_connected_layer_output = self.fc_layer(inputs=input_x)
    ghost_batch_norm_output = self.ghost_batch_norm(
        input_x=fully_connected_layer_output, train=train
    )
    return jnp.multiply(
        ghost_batch_norm_output[:, : self.output_dimensions],
        nn.sigmoid(ghost_batch_norm_output[:, self.output_dimensions :]),
    )


class GatedLinearUnitBlock(nn.Module):
  """Implementation of a Gated Linear Unit Block.

  Attributes:
    input_dimensions: The number of feature columns in the input table.
    output_dimensions: The number of possible actions.
    shared_layers: Fully-connected Dense layers.
    number_of_glu_layers: A number of GLU layers.
    first: Indicates if the first layer of the block has no scale
      multiplication.
    virtual_batch_size: Batch size for Ghost Batch Normalization.
    momentum: Momentum value for the BatchNorm layer.
    model_data_type: Data type to use within the model.
  """

  input_dimensions: int
  output_dimensions: int
  shared_layers: tuple[nn.Module, ...] | None = None
  number_of_glu_layers: int = 2
  first: bool = False
  virtual_batch_size: int = 128
  momentum: float = 0.98
  model_data_type: jnp.dtype = jnp.float32

  def setup(self) -> None:
    """Sets up the module definition when it's called."""

    if self.shared_layers:
      self.glu_layers = [
          GatedLinearUnitLayer(
              self.input_dimensions,
              self.output_dimensions,
              fully_connected_layer=self.shared_layers[i],
              virtual_batch_size=self.virtual_batch_size,
              momentum=self.momentum,
              model_data_type=self.model_data_type,
          )
          for i in range(self.number_of_glu_layers)
      ]
    else:
      self.glu_layers = [
          GatedLinearUnitLayer(
              self.input_dimensions,
              self.output_dimensions,
              fully_connected_layer=None,
              virtual_batch_size=self.virtual_batch_size,
              momentum=self.momentum,
              model_data_type=self.model_data_type,
          )
          for _ in range(self.number_of_glu_layers)
      ]


  @nn.compact
  def __call__(
      self, *, input_x: jnp.ndarray, train: bool = False
  ) -> jnp.ndarray:
    """Forward pass implementation for the layer.

    Args:
      input_x: An input array to be processed by the GLU Block.
      train: An indicator whether the pass is during training or not.

    Returns:
      The output of the forward pass.
    """
    scale = jnp.sqrt(jnp.array([0.5]))
    if self.first:
      input_x = self.glu_layers[0](input_x=input_x, train=train)
      layers_left = range(1, self.number_of_glu_layers)
    else:
      layers_left = range(self.number_of_glu_layers)

    for glu_id in layers_left:
      input_x = jnp.add(
          input_x, self.glu_layers[glu_id](input_x=input_x, train=train)
      )
      input_x *= scale
    return input_x


class Identity(nn.Module):
  """Implementation of the identity layer."""

  @nn.compact
  def __call__(
      self, *, input_x: jnp.ndarray, train: bool = False
  ) -> jnp.ndarray:
    """Forward pass implementation for the layer.

    Args:
      input_x: An input array to be passed through the Identiy layer.
      train: An indicator whether the pass is during training or not.

    Returns:
      The output of the forward pass.
    """
    del train
    return input_x


class FeatureTransformer(nn.Module):
  """Feature transformer layer.

  Attributes:
    input_dimensions: The number of feature columns in the input table.
    output_dimensions: The number of possible actions.
    number_of_independent_glu_layers: A number of independent GLU layers.
    shared_layers: Fully-connected Dense layers.
    virtual_batch_size: Batch size for Ghost Batch Normalization.
    momentum: Momentum value for the BatchNorm layer.
    model_data_type: Data type to use within the model.
  """

  input_dimensions: int
  output_dimensions: int
  number_of_independent_glu_layers: int
  shared_layers: tuple[nn.Module, ...] | None = None
  virtual_batch_size: int = 128
  momentum: float = 0.98
  model_data_type: jnp.dtype = jnp.float32

  def setup(self) -> None:
    """Sets up the module definition when it's called."""
    if not self.shared_layers:
      self.shared = Identity()
      is_first = True
    else:
      self.shared = GatedLinearUnitBlock(
          input_dimensions=self.input_dimensions,
          output_dimensions=self.output_dimensions,
          first=True,
          shared_layers=self.shared_layers,
          number_of_glu_layers=len(self.shared_layers),
          virtual_batch_size=self.virtual_batch_size,
          momentum=self.momentum,
          model_data_type=self.model_data_type,
      )
      is_first = False
    if self.number_of_independent_glu_layers == 0:
      self.specifics = Identity()
    else:
      spec_input_dim = (
          self.input_dimensions if is_first else self.output_dimensions
      )
      self.specifics = GatedLinearUnitBlock(
          input_dimensions=spec_input_dim,
          output_dimensions=self.output_dimensions,
          first=is_first,
          number_of_glu_layers=self.number_of_independent_glu_layers,
          virtual_batch_size=self.virtual_batch_size,
          momentum=self.momentum,
          model_data_type=self.model_data_type,
      )

  @nn.compact
  def __call__(
      self, *, input_x: jnp.ndarray, train: bool = False
  ) -> jnp.ndarray:
    """Forward pass implementation for the layer.

    Args:
      input_x: An input array to be processed by the FeatureTransformer layer.
      train: An indicator whether the pass is during training or not.

    Returns:
      The output of the forward pass.
    """
    shared_layers_output = self.shared(input_x=input_x, train=train)
    return self.specifics(input_x=shared_layers_output, train=train)


def _reshape_to_broadcast(
    *, indexes: jnp.ndarray, shape: tuple[int, ...], axis: int
) -> jnp.ndarray:
  """Utility function for the Sparsemax activation.

  Args:
    indexes: Indices from a specified axis.
    shape: Unmodified array shape.
    axis: An axis for the computation.

  Returns:
    The reshaped array for broadcasting.
  """
  new_shape = [1] * len(shape)
  new_shape[axis] = shape[axis]
  return jnp.reshape(indexes, new_shape)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
@functools.partial(jax.jit, static_argnums=(1,))
def _sparsemax(input_x: jnp.ndarray, axis: int) -> jnp.ndarray:
  """Utility function for the Sparsemax activation.

  The function defines the custom differentiation rule for the Sparsemax
  activation.

  Args:
    input_x: An input to use int the computation.
    axis: An axis for the computation.

  Returns:
    An output transformed with the Sparsemax activation function.
  """
  idxs = jnp.arange(input_x.shape[axis]) + 1
  idxs = _reshape_to_broadcast(indexes=idxs, shape=input_x.shape, axis=axis)
  sorted_x = jnp.flip(jax.lax.sort(input_x, dimension=axis), axis=axis)# jnp-type
  cum = jnp.cumsum(sorted_x, axis=axis)
  filtered_cum = jnp.sum(
      jnp.where(1 + sorted_x * idxs > cum, 1, 0), axis=axis, keepdims=True
  )
  threshold = (
      jnp.take_along_axis(cum, filtered_cum - 1, axis=axis) - 1
  ) / filtered_cum
  return jnp.maximum(input_x - threshold, 0)


@_sparsemax.defjvp
@functools.partial(jax.jit, static_argnums=(0,))
def _sparsemax_jvp(
    axis: int, primals: jnp.ndarray, tangents: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Utility function for the Sparsemax activation.

  Args:
    axis: An axis for the computation.
    primals: Inputs to a primal function of type a.
    tangents: Corresponding tanget inputs of type T(a).

  Returns:
    Transformed input and gradients.
  """
  input_x = primals[0]
  tangent_input_x = tangents[0]
  entmax_projection = _sparsemax(input_x, axis)
  auxiliary_mask = jnp.where(entmax_projection > 0, 1, 0)
  tangent_y = tangent_input_x * auxiliary_mask
  gradients = jnp.sum(tangent_y, axis=axis) / jnp.sum(auxiliary_mask, axis=axis)
  tangent_y = tangent_y - jnp.expand_dims(gradients, axis) * auxiliary_mask
  return entmax_projection, tangent_y


def sparsemax(*, input_x: jnp.ndarray, axis: int) -> jnp.ndarray:
  """Sparsemax axtivation function.

  See: https://arxiv.org/abs/1602.02068
  See: https://github.com/deep-spin/entmax-jax

  Args:
    input_x: An input to use in the computation.
    axis: An axis for the computation.

  Returns:
    An output transformed with the Sparsemax activation function.
  """
  return _sparsemax(input_x, axis)


class AttentiveTransformer(nn.Module):
  """Attentive transformer layer.

  Attributes:
    input_dimensions: The number of feature columns in the input table.
    output_dimensions: The number of possible actions.
    virtual_batch_size: Batch size for Ghost Batch Normalization.
    momentum: Momentum value for the BatchNorm layer.
    model_data_type: Data type to use within the model.
  """

  input_dimensions: int
  output_dimensions: int
  virtual_batch_size: int = 128
  momentum: float = 0.98
  mask_type: str = "sparsemax"
  model_data_type: jnp.dtype = jnp.float32

  def setup(self) -> None:
    """Sets up the module definition when it's called."""
    gain_value = jnp.sqrt(
        (self.input_dimensions + self.output_dimensions)
        / jnp.sqrt(4 * self.input_dimensions)
    )
    kernel_init = nn.initializers.variance_scaling(
        scale=gain_value, mode="fan_avg", distribution="truncated_normal"
    )
    self.fully_connected_layer = nn.Dense(
        self.output_dimensions,
        use_bias=False,
        kernel_init=kernel_init,
        dtype=self.model_data_type,
    )
    self.ghost_batch_norm_output = GhostBatchNormalization(
        virtual_batch_size=self.virtual_batch_size,
        momentum=self.momentum,
        model_data_type=self.model_data_type,
    )

  @nn.compact
  def __call__(
      self, *, priors: jnp.ndarray, input_x: jnp.ndarray, train: bool = False
  ) -> jnp.ndarray:
    """Forward pass implementation for the layer.

    Args:
      priors: An array with priors scale information.
      input_x: An input array to be processed by the AttentiveTransformer layer.
      train: An indicator whether the pass is during training or not.

    Returns:
      The output of the forward pass.

    Raises:
      NotImplementedError: An error when trying to use an unavailble mask type.
    """
    if self.mask_type != "sparsemax":
      raise NotImplementedError("Only sparsemax is implemented.")
    fully_connected_layer_output = self.fully_connected_layer(inputs=input_x)
    ghost_batch_norm_output = self.ghost_batch_norm_output(
        input_x=fully_connected_layer_output, train=train
    )
    multiplification_output = jnp.multiply(ghost_batch_norm_output, priors)
    return sparsemax(input_x=multiplification_output, axis=-1)


class TabNetEncoder(nn.Module):
  """TabNet Encoder layer.

  Attributes:
    input_dimensions: The number of feature columns in the input table.
    output_dimensions: The number of possible actions.
    prediction_layer_dimension: Dimension of the prediction layer (usually
      between 4 and 64).
    attention_layer_dimension: Dimension of the attention layer (usually between
      4 and 64).
    successive_network_steps: Number of successive steps in the network (usually
      between 3 and 10).
    attention_gamma: Float above 1, scaling factor for attention updates
      (usually between 1.0 to 2.0).
    independent_glu_layers: A number of independent GLU layers in each GLU
      block.
    shared_glu_layers: A number of shared GLU layer in each GLU block.
    epsilon: a factor to avoid jnp.log(0).
    virtual_batch_size: Batch size for Ghost Batch Normalization.
    momentum: Momentum value for the BatchNorm layer.
    mask_type: The masking function to use.
    model_data_type: Data type to use within the model.
  """

  input_dimensions: int
  output_dimensions: int | tuple[int, ...]
  prediction_layer_dimension: int = 8
  attention_layer_dimension: int = 8
  successive_network_steps: int = 3
  attention_gamma: float = 1.3
  independent_glu_layers: int = 2
  shared_glu_layers: int = 2
  epsilon: float = 1e-15
  virtual_batch_size: int = 128
  momentum: float = 0.98
  mask_type: str = "sparsemax"
  model_data_type: jnp.dtype = jnp.float32

  def setup(self) -> None:
    """Sets up the module definition when it's called."""

    if self.shared_glu_layers == 0:
      shared_feat_transform = ()
    else:
      shared_feat_transform = [
          nn.Dense(
              2
              * (
                  self.prediction_layer_dimension
                  + self.attention_layer_dimension
              ),
              use_bias=False,
              dtype=self.model_data_type,
          )
          for _ in range(self.shared_glu_layers)
      ]
      shared_feat_transform = tuple(shared_feat_transform)

    self.initial_splitter = FeatureTransformer(
        input_dimensions=self.input_dimensions,
        output_dimensions=self.prediction_layer_dimension
        + self.attention_layer_dimension,
        shared_layers=shared_feat_transform,
        number_of_independent_glu_layers=self.independent_glu_layers,
        virtual_batch_size=self.virtual_batch_size,
        momentum=self.momentum,
        model_data_type=self.model_data_type,
    )
    self.feat_transformers = [
        FeatureTransformer(
            input_dimensions=self.input_dimensions,
            output_dimensions=self.prediction_layer_dimension
            + self.attention_layer_dimension,
            shared_layers=shared_feat_transform,
            number_of_independent_glu_layers=self.independent_glu_layers,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            model_data_type=self.model_data_type,
        )
    ] * self.successive_network_steps
    self.att_transformers = [
        AttentiveTransformer(
            input_dimensions=self.attention_layer_dimension,
            output_dimensions=self.input_dimensions,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            mask_type=self.mask_type,
            model_data_type=self.model_data_type,
        )
    ] * self.successive_network_steps

  @nn.compact
  def __call__(# jnp-array
      self,
      *,
      input_x: jnp.ndarray,
      prior: jnp.ndarray = None,
      forward_masks: bool = False,
      train: bool = False,
  ) -> (
      tuple[list[jnp.ndarray], list[jnp.ndarray]]
      | tuple[jnp.ndarray, dict[int, jnp.ndarray]]
  ):
    """Forward pass implementation for the layer.

    Args:
      input_x: An input array to be processed by the TabNetEncoder layer.
      prior: An array with prior scale information.
      forward_masks: An indicator if the forward_masks process should be called.
      train: An indicator whether the pass is during training or not.

    Returns:
      The output of the forward pass.
    """
    batch_norm = nn.BatchNorm(
        momentum=0.99, use_running_average=not train, dtype=self.model_data_type
    )
    batch_normalization_output = batch_norm(x=input_x)
    if not forward_masks:
      if prior is None:
        prior = jnp.ones(
            batch_normalization_output.shape, dtype=self.model_data_type
        )
      attentive_transformer_loss = []
      feature_transformer_output = self.initial_splitter(
          input_x=batch_normalization_output, train=train
      )[:, self.prediction_layer_dimension :]
      steps_output = []
      for step in range(self.successive_network_steps):
        attentive_transformer_layer_output = self.att_transformers[step](
            priors=prior, input_x=feature_transformer_output, train=train
        )
        attentive_transformer_loss.append(
            jnp.mean(
                jnp.sum(
                    jnp.multiply(
                        attentive_transformer_layer_output,
                        jnp.log(
                            attentive_transformer_layer_output + self.epsilon
                        ),
                    ),
                    axis=1,
                )
            )
        )
        prior = jnp.multiply(
            self.attention_gamma - attentive_transformer_layer_output, prior
        )
        masked_x = jnp.multiply(
            attentive_transformer_layer_output, batch_normalization_output
        )
        output = self.feat_transformers[step](input_x=masked_x, train=train)
        post_activation_out = nn.relu(
            output[:, : self.prediction_layer_dimension]
        )
        steps_output.append(post_activation_out)
        feature_transformer_output = output[
            :, self.prediction_layer_dimension :
        ]

      return steps_output, attentive_transformer_loss

    prior = jnp.ones(
        batch_normalization_output.shape, dtype=self.model_data_type
    )
    attentive_transformer_explain = jnp.zeros(
        batch_normalization_output.shape, dtype=self.model_data_type
    )
    feature_transformer_output = self.initial_splitter(
        input_x=batch_normalization_output, train=train
    )[:, self.prediction_layer_dimension :]
    masks = {}

    for step in range(self.successive_network_steps):
      attentive_transformer_layer_output = self.att_transformers[step](
          priors=prior, input_x=feature_transformer_output
      )
      masks[step] = attentive_transformer_layer_output
      prior = jnp.multiply(
          self.attention_gamma - attentive_transformer_layer_output, prior
      )
      masked_x = jnp.multiply(
          attentive_transformer_layer_output, batch_normalization_output
      )
      output = self.feat_transformers[step](input_x=masked_x, train=train)
      post_activation_out = nn.relu(
          output[:, : self.prediction_layer_dimension]
      )
      step_importance = jnp.sum(post_activation_out, axis=1)
      attentive_transformer_explain += jnp.multiply(
          attentive_transformer_layer_output,
          jnp.expand_dims(step_importance, axis=1),
      )
      feature_transformer_output = output[:, self.prediction_layer_dimension :]
    return attentive_transformer_explain, masks


class TabNetDecoder(nn.Module):
  """TabNet Decoder layer.

  Attributes:
    input_dimensions: The number of feature columns in the input table.
    prediction_layer_dimension: Dimension of the prediction layer (usually
      between 4 and 64).
    successive_network_steps: Number of successive steps in the network (usually
      between 3 and 10).
    independent_glu_layers: A number of independent GLU layer in each GLU block.
    shared_glu_layers: A number of shared GLU layers in each GLU block.
    virtual_batch_size: Batch size for Ghost Batch Normalization.
    momentum: Momentum value for the BatchNorm layer.
    model_data_type: Data type to use within the model.
  """

  input_dimensions: int
  prediction_layer_dimension: int = 8
  successive_network_steps: int = 3
  independent_glu_layers: int = 1
  shared_glu_layers: int = 1
  virtual_batch_size: int = 128
  momentum: float = 0.98
  model_data_type: jnp.dtype = jnp.float32

  def setup(self) -> None:
    """Sets up the module definition when it's called."""
    if self.shared_glu_layers == 0:
      shared_feat_transform = ()
    else:
      shared_feat_transform = [
          nn.Dense(2 * self.prediction_layer_dimension, use_bias=False)
      ] * self.shared_glu_layers
      shared_feat_transform = tuple(shared_feat_transform)
    self.feat_transformers = [
        FeatureTransformer(
            input_dimensions=self.prediction_layer_dimension,
            output_dimensions=self.prediction_layer_dimension,
            shared_layers=shared_feat_transform,
            number_of_independent_glu_layers=self.independent_glu_layers,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            model_data_type=self.model_data_type,
        )
    ] * self.successive_network_steps
    gain_value = jnp.sqrt(
        (self.input_dimensions + self.prediction_layer_dimension)
        / jnp.sqrt(4 * self.input_dimensions)
    )
    kernel_init = nn.initializers.variance_scaling(
        scale=gain_value, mode="fan_avg", distribution="truncated_normal"
    )
    self.reconstruction_layer = nn.Dense(
        self.input_dimensions,
        use_bias=False,
        kernel_init=kernel_init,
        dtype=self.model_data_type,
    )

  @nn.compact
  def __call__(
      self, *, input_x: list[float], train: bool = False
  ) -> jnp.ndarray:
    """Forward pass implementation for the layer.

    Args:
      input_x: An input array.
      train: An indicator whether the pass is during training or not.

    Returns:
      The output of the forward pass.
    """
    result = 0.0
    for step_nb, step_output in enumerate(input_x):
      feature_transformer_output = self.feat_transformers[step_nb](# jnp-array
          input_x=step_output, train=train
      )
      result = jnp.add(result, feature_transformer_output)
    return self.reconstruction_layer(inputs=result)


class TabNetPretraining(nn.Module):
  """TabNet Pretraining layer.

  Attributes:
    input_dimensions: The number of feature columns in the input table.
    pretraining_ratio: The ratio of data to randomly obfuscate. between 4 and
      64).
    attention_layer_dimension: Dimension of the attention layer (usually between
      4 and 64).
    successive_network_steps: Number of successive steps in the network (usually
      between 3 and 10).
    attention_gamma: Float above 1, scaling factor for attention updates
      (usually between 1.0 to 2.0).
    categorical_indexes: Indices of categorical columns in the input table.
    categorical_indexes: Indices of categorical columns in the input table.
    categorical_dimensions: The number of dimensions in each of the categorical
      columns of the input table.
    categorical_embedding_dimensions: The size of the embedding of categorical
      features if int, all categorical features will have same embedding size if
      list of int, every corresponding feature will have specific size.
    independent_glu_layers: A number of independent GLU layers in each GLU
      block.
    shared_glu_layers: A number of shared GLU layers in each GLU block.
    epsilon: a constant to avoid jnp.log(0).
    virtual_batch_size: Batch size for Ghost Batch Normalization.
    momentum: Momentum value for the BatchNorm layer.
    mask_type: The masking function to use.
    shared_decoder_layers: A number of shared decoder layers in each GLU block
      in the decoder.
    independent_decoder_layers: A number of independent decoder layers in each
      GLU block in the decoder.
    model_data_type: Data type to use within the model.
  """

  input_dimensions: int
  pretraining_ratio: float = 0.2
  prediction_layer_dimension: int = 8
  attention_layer_dimension: int = 8
  successive_network_steps: int = 3
  attention_gamma: float = 1.3
  categorical_dimensions: tuple[int, ...] | None = None
  categorical_indexes: tuple[int, ...] | None = None
  categorical_embedding_dimensions: tuple[int, ...] | None = None
  independent_glu_layers: int = 2
  shared_glu_layers: int = 2
  epsilon: float = 1e-15
  virtual_batch_size: int = 128
  momentum: float = 0.98
  mask_type: str = "sparsemax"
  shared_decoder_layers: int = 1
  independent_decoder_layers: int = 1
  model_data_type: jnp.dtype = jnp.float32

  def setup(self) -> None:
    """Sets up the module definition when it's called.

    Raises:
      ValueError: An error when successive_network_steps is less or equal to 0.
        Or when independent_glu_layers and shared_glu_layers are equal 0.
    """
    if self.successive_network_steps <= 0:
      raise ValueError("successive_network_steps should be a positive integer.")
    if self.independent_glu_layers == 0 and self.shared_glu_layers == 0:
      raise ValueError(
          "shared_glu_layers and independent_glu_layers can not be both zero."
      )
    categorical_dimensions = self.categorical_dimensions or ()
    categorical_indexes = self.categorical_indexes or ()
    self.embedder = EmbeddingGenerator(
        self.input_dimensions,
        categorical_dimensions,
        categorical_indexes,
        self.categorical_embedding_dimensions,
        self.model_data_type,
    )
    self.post_embed_dim = self.embedder.post_embedding_dimension
    self.masker = RandomObfuscator(self.pretraining_ratio)
    self.encoder = TabNetEncoder(
        input_dimensions=self.post_embed_dim,
        output_dimensions=self.post_embed_dim,
        prediction_layer_dimension=self.prediction_layer_dimension,
        attention_layer_dimension=self.attention_layer_dimension,
        successive_network_steps=self.successive_network_steps,
        attention_gamma=self.attention_gamma,
        independent_glu_layers=self.independent_glu_layers,
        shared_glu_layers=self.shared_glu_layers,
        epsilon=self.epsilon,
        virtual_batch_size=self.virtual_batch_size,
        momentum=self.momentum,
        mask_type=self.mask_type,
        model_data_type=self.model_data_type,
    )
    self.decoder = TabNetDecoder(
        input_dimensions=self.post_embed_dim,
        prediction_layer_dimension=self.prediction_layer_dimension,
        successive_network_steps=self.successive_network_steps,
        independent_glu_layers=self.independent_decoder_layers,
        shared_glu_layers=self.shared_decoder_layers,
        virtual_batch_size=self.virtual_batch_size,
        momentum=self.momentum,
        model_data_type=self.model_data_type,
    )

  @nn.compact
  def __call__(
      self, *, input_x: jnp.ndarray, train: bool = False
  ) -> tuple[jnp.ndarray, ...]:
    """Forward pass implementation for the layer.

    Args:
      input_x: An input array to be processed by the TabNetPretraining layer.
      train: An indicator whether the pass is during training or not.

    Returns:
      The output of the forward pass.
    """
    embedded_x = self.embedder(input_x=input_x)
    if train:
      masked_x, obf_vars = self.masker(input_x=embedded_x)
      prior = 1 - obf_vars
      steps_out, _ = self.encoder(input_x=masked_x, prior=prior, train=train)
      result = self.decoder(input_x=steps_out, train=train)# jnp-array
      return result, embedded_x, obf_vars
    steps_out, _ = self.encoder(input_x=embedded_x, train=train)
    result = self.decoder(input_x=steps_out, train=train)# jnp-array
    return result, embedded_x, jnp.ones(embedded_x.shape)


class TabNetCoreTraining(nn.Module):
  """TabNet Core Training layer.

  The TabNet architecture used for training and NOT pretraining.

  Attributes:
    input_dimensions: The number of feature columns in the input table.
    output_dimensions: The number of possible actions.
    prediction_layer_dimension: Dimension of the prediction layer (usually
      between 4 and 64).
    attention_layer_dimension: Dimension of the attention layer (usually between
      4 and 64).
    successive_network_steps: Number of successive steps in the network (usually
      between 3 and 10).
    attention_gamma: Float above 1, scaling factor for attention updates
      (usually between 1.0 to 2.0).
    independent_glu_layers: A number of independent GLU layer in each GLU block.
    shared_glu_layers: A number of shared GLU layer in each GLU block.
    epsilon: A factor to avoid jnp.log(0).
    virtual_batch_size: Batch size for Ghost Batch Normalization.
    momentum: Momentum value for the BatchNorm layer.
    mask_type: The masking function to use.
    model_data_type: Data type to use within the model.
  """

  input_dimensions: int
  output_dimensions: int | tuple[int, ...]
  prediction_layer_dimension: int = 8
  attention_layer_dimension: int = 8
  successive_network_steps: int = 3
  attention_gamma: float = 1.3
  independent_glu_layers: int = 2
  shared_glu_layers: int = 2
  epsilon: float = 1e-15
  virtual_batch_size: int = 128
  momentum: float = 0.98
  mask_type: str = "sparsemax"
  model_data_type: jnp.dtype = jnp.float32

  def setup(self) -> None:
    """Sets up the module definition when it's called."""

    self.encoder = TabNetEncoder(
        input_dimensions=self.input_dimensions,
        output_dimensions=self.output_dimensions,
        prediction_layer_dimension=self.prediction_layer_dimension,
        attention_layer_dimension=self.attention_layer_dimension,
        successive_network_steps=self.successive_network_steps,
        attention_gamma=self.attention_gamma,
        independent_glu_layers=self.independent_glu_layers,
        shared_glu_layers=self.shared_glu_layers,
        epsilon=self.epsilon,
        virtual_batch_size=self.virtual_batch_size,
        momentum=self.momentum,
        mask_type=self.mask_type,
        model_data_type=self.model_data_type,
    )

    if isinstance(self.output_dimensions, tuple):
      n_preds = [self.prediction_layer_dimension] * len(self.output_dimensions)
      gain_values = [
          jnp.sqrt((i + o) / jnp.sqrt(4 * i))
          for i, o in zip(n_preds, self.output_dimensions)
      ]
      kernel_inits = [
          nn.initializers.variance_scaling(
              scale=g, mode="fan_avg", distribution="truncated_normal"
          )
          for g in gain_values
      ]
      self.multi_task_mappings = [
          nn.Dense(o, use_bias=False, kernel_init=k, dtype=self.model_data_type)
          for o, k in zip(self.output_dimensions, kernel_inits)
      ]
    else:
      gain_value = jnp.sqrt(
          (self.prediction_layer_dimension + self.output_dimensions)
          / jnp.sqrt(4 * self.prediction_layer_dimension)
      )
      kernel_init = nn.initializers.variance_scaling(
          scale=gain_value, mode="fan_avg", distribution="truncated_normal"
      )
      self.final_mapping = nn.Dense(
          self.output_dimensions,
          use_bias=False,
          kernel_init=kernel_init,
          dtype=self.model_data_type,
      )
    self.value_layer = nn.Dense(
        features=1, name="value", dtype=self.model_data_type
    )

  @nn.compact
  def __call__(
      self,
      *,
      input_x: jnp.ndarray,
      forward_masks: bool = False,
      train: bool = False,
  ) -> (
      tuple[list[jnp.ndarray], list[jnp.ndarray]]
      | tuple[jnp.ndarray, dict[int, jnp.ndarray]]
      | tuple[jnp.ndarray, jnp.ndarray, float]
  ):
    """Forward pass implementation for the layer.

    Args:
      input_x: An input array.
      forward_masks: An indicator if the forward_masks process should be called.
      train: An indicator whether the pass is during training or not.

    Returns:
      The output of the forward pass.
    """
    if not forward_masks:
      steps_output, attentive_transformer_loss = self.encoder(
          input_x=input_x, train=train
      )
      result = jnp.sum(jnp.stack(steps_output, axis=0), axis=0)
      attentive_transformer_loss = jnp.mean(
          jnp.stack(attentive_transformer_loss, axis=0), axis=0
      )
      if isinstance(self.output_dimensions, tuple):
        logits = []
        for task_mapping in self.multi_task_mappings:
          logits.append(task_mapping(inputs=result))
        policy_log_probabilities = [nn.log_softmax(i) for i in logits]
      else:
        logits = self.final_mapping(inputs=result)
        policy_log_probabilities = nn.log_softmax(logits)
      value = self.value_layer(inputs=result)
      return (policy_log_probabilities, value, attentive_transformer_loss)# jnp-array
    return self.encoder(input_x=input_x, train=train, forward_masks=True)


class TabNet(nn.Module):
  """TabNet model.

  Attributes:
    output_dimensions: The number of possible actions.
    prediction_layer_dimension: Dimension of the prediction layer (usually
      between 4 and 64).
    attention_layer_dimension: Dimension of the attention layer (usually between
      4 and 64).
    successive_network_steps: Number of successive steps in the network (usually
      between 3 and
    categorical_indexes: Indices of categorical columns in the input table.
    categorical_dimensions: The number of dimensions in each of the categorical
    categorical_embedding_dimensions: The size of the embedding of categorical
      features if int, all categorical features will have same embedding size if
      list of int, every corresponding feature will have specific size.
    independent_glu_layers: A number of independent GLU layers in each GLU
      block.
    shared_glu_layers: A number of shared GLU layer in each GLU block.
    epsilon: A factor to avoid jnp.log(0).
    virtual_batch_size: Batch size for Ghost Batch Normalization.
    momentum: Momentum value for the BatchNorm layer.
    mask_type: The masking function to use.
    model_data_type: Data type to use within the model.
  """

  input_dimensions: int
  output_dimensions: int | tuple[int, ...]
  prediction_layer_dimension: int = 8
  attention_layer_dimension: int = 8
  successive_network_steps: int = 3
  attention_gamma: float = 1.3
  categorical_dimensions: tuple[int, ...] | None = None
  categorical_indexes: tuple[int, ...] | None = None
  categorical_embedding_dimensions: tuple[int, ...] | None = None
  independent_glu_layers: int = 2
  shared_glu_layers: int = 2
  epsilon: float = 1e-15
  virtual_batch_size: int = 128
  momentum: float = 0.98
  mask_type: str = "sparsemax"
  model_data_type: jnp.dtype = jnp.float32

  def setup(self) -> None:
    """Sets up the module definition when it's called.

    Raises:
      ValueError: An error when successive_network_steps is less or equal to 0.
        Or when independent_glu_layers and shared_glu_layers are equal 0.
    """
    if self.successive_network_steps <= 0:
      raise ValueError("successive_network_steps should be a positive integer.")
    if self.independent_glu_layers == 0 and self.shared_glu_layers == 0:
      raise ValueError(
          "shared_glu_layers and independent_glu_layers can not be both zero."
      )

    categorical_dimensions = (
        [] if not self.categorical_dimensions else self.categorical_dimensions
    )
    categorical_indexes = (
        [] if not self.categorical_indexes else self.categorical_indexes
    )
    self.embedder = EmbeddingGenerator(
        self.input_dimensions,
        categorical_dimensions,
        categorical_indexes,
        self.categorical_embedding_dimensions,
        self.model_data_type,
    )
    self.post_embed_dim = self.embedder.post_embedding_dimension
    self.tabnet = TabNetCoreTraining(
        input_dimensions=self.post_embed_dim,
        output_dimensions=self.output_dimensions,
        prediction_layer_dimension=self.prediction_layer_dimension,
        attention_layer_dimension=self.attention_layer_dimension,
        successive_network_steps=self.successive_network_steps,
        attention_gamma=self.attention_gamma,
        independent_glu_layers=self.independent_glu_layers,
        shared_glu_layers=self.shared_glu_layers,
        epsilon=self.epsilon,
        virtual_batch_size=self.virtual_batch_size,
        momentum=self.momentum,
        mask_type=self.mask_type,
        model_data_type=self.model_data_type,
    )

  @nn.compact
  def __call__(
      self,
      *,
      input_x: jnp.ndarray,
      forward_masks: bool = False,
      train: bool = False,
  ) -> (
      tuple[list[jnp.ndarray], list[jnp.ndarray]]
      | tuple[jnp.ndarray, dict[int, jnp.ndarray]]
      | tuple[jnp.ndarray, jnp.ndarray, float]
  ):
    """Forward pass implementation for the layer.

    Args:
      input_x: An input array to be processed by the TabNet layer.
      forward_masks: An indicator if the forward_masks process should be called.
      train: An indicator whether the pass is during training or not.

    Returns:
      The output of the forward pass.
    """
    if not forward_masks:
      if input_x.dtype != self.model_data_type:
        input_x = jnp.array(input_x, dtype=self.model_data_type)
      embedder_output = self.embedder(input_x=input_x)
      return self.tabnet(input_x=embedder_output, train=train)
    embedder_output = self.embedder(input_x=input_x)
    return self.tabnet(input_x=embedder_output, train=train, forward_masks=True)


class TabNetAgent(base_agent.BaseAgent):
  """Reinforcement Learning agent based on TabNet architecture."""

  def build_flax_module(self):
    """Build the TabNet architecture."""
    return TabNet(
        input_dimensions=self.hyperparameters.input_dimensions,
        output_dimensions=self.hyperparameters.action_space,
        categorical_indexes=self.hyperparameters.categorical_indexes,
        categorical_dimensions=self.hyperparameters.categorical_dimensions,
        prediction_layer_dimension=self.hyperparameters.prediction_layer_dimension,
        attention_layer_dimension=self.hyperparameters.attention_layer_dimension,
        successive_network_steps=self.hyperparameters.successive_network_steps,
        attention_gamma=self.hyperparameters.attention_gamma,
        categorical_embedding_dimensions=self.hyperparameters.categorical_embedding_dimensions,
        independent_glu_layers=self.hyperparameters.independent_glu_layers,
        shared_glu_layers=self.hyperparameters.shared_glu_layers,
        epsilon=self.hyperparameters.epsilon,
        virtual_batch_size=self.hyperparameters.virtual_batch_size,
        momentum=self.hyperparameters.momentum,
        mask_type=self.hyperparameters.mask_type,
        model_data_type=self.hyperparameters.model_data_type,
    )
