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

"""Tests for utility functions in tabnet.py."""
from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import optax

from optimus.agent_lib import base_agent
from optimus.agent_lib import tabnet

jax.config.update("jax_threefry_partitionable", False)

_TEST_TABNET_HYPERPARAMETERS = config_dict.ConfigDict(
    dict(
        action_space=1,
        input_dimensions=4,
        output_dimensions=1,
        categorical_indexes=None,
        categorical_dimensions=None,
        attention_gamma=1.3,
        prediction_layer_dimension=8,
        attention_layer_dimension=8,
        successive_network_steps=3,
        categorical_embedding_dimensions=[1],
        independent_glu_layers=2,
        shared_glu_layers=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.98,
        mask_type="sparsemax",
        shared_decoder_layers=1,
        independent_decoder_layers=1,
        model_data_type="float32",
        batch_size=1,
        value_function_coefficient=0.5,
        entropy_coefficient=0.02,
        lambda_sparse=1e-3,
        action_space_length=1,
        replace_nans_in_prediction=True,
    )
)

_TEST_GAIN_VALUE = (8.0**0.5) / (4.0**0.5)
_TEST_KERNEL_INIT = nn.initializers.variance_scaling(
    scale=_TEST_GAIN_VALUE, mode="fan_avg", distribution="truncated_normal"
)


class EmbeddingGeneratorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="categorical_dimensions_or_categorical_indexes_is_none",
          categorical_dimensions=(1, 2),
          categorical_indexes=None,
          categorical_embedding_dimensions=(1,),
          error_message=(
              "categorical_dimensions and categorical_indexes are required"
          ),
      ),
      dict(
          testcase_name=(
              "categorical_dimensions_and_categorical_indexes_of"
              "_different_length"
          ),
          categorical_dimensions=(1, 2),
          categorical_indexes=(1,),
          categorical_embedding_dimensions=(1,),
          error_message="categorical_dimensions and categorical_indexes must",
      ),
      dict(
          testcase_name=(
              "categorical_dimensions_and_cat_emb_dims_have_different_length"
          ),
          categorical_dimensions=(1, 2),
          categorical_indexes=(1, 2),
          categorical_embedding_dimensions=(
              1,
              2,
              3,
              4,
          ),
          error_message=(
              "categorical_embedding_dimensions and categorical_dimensions must"
          ),
      ),
  )
  def test_embedding_generator_value_error(
      self,
      categorical_dimensions,
      categorical_indexes,
      categorical_embedding_dimensions,
      error_message,
  ):
    with self.assertRaisesRegex(ValueError, error_message):
      embedding_generator = tabnet.EmbeddingGenerator(
          input_dimensions=2,
          categorical_dimensions=categorical_dimensions,
          categorical_indexes=categorical_indexes,
          categorical_embedding_dimensions=categorical_embedding_dimensions,
      )
      embedding_generator.init(
          rngs={"params": jax.random.PRNGKey(0)},
          input_x=jnp.array([[1, 2], [1, 2]], dtype=jnp.float32),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="without_categorical_dimensions_or_categorical_indexes",
          categorical_dimensions=None,
          categorical_indexes=None,
          categorical_embedding_dimensions=(1,),
      ),
      dict(
          testcase_name=(
              "with_categorical_dimensions_and_no_categorical_indexes"
          ),
          categorical_dimensions=(1,),
          categorical_indexes=(2,),
          categorical_embedding_dimensions=(1,),
      ),
      dict(
          testcase_name="categorical_embedding_dimensions_as_tuple",
          categorical_dimensions=(1,),
          categorical_indexes=(2,),
          categorical_embedding_dimensions=(1,),
      ),
  )
  def test_embedding_generator(
      self,
      categorical_dimensions,
      categorical_indexes,
      categorical_embedding_dimensions,
  ):
    input_x = jnp.array([[1, 2], [1, 2]], dtype=jnp.float32)
    embedding_generator = tabnet.EmbeddingGenerator(
        input_dimensions=2,
        categorical_dimensions=categorical_dimensions,
        categorical_indexes=categorical_indexes,
        categorical_embedding_dimensions=categorical_embedding_dimensions,
    )
    parameters = embedding_generator.init(
        rngs={"params": jax.random.PRNGKey(0)}, input_x=input_x
    )
    self.assertEqual(
        embedding_generator.apply(
            variables={"params": parameters}, input_x=input_x
        ).tolist(),
        [[1.0, 2.0], [1.0, 2.0]],
    )


class RandomObfuscatorTest(absltest.TestCase):

  def test_random_obfuscator(self):
    rng_key, bernoulli_key = jax.random.split(jax.random.PRNGKey(0))
    rngs_keys = {"params": rng_key, "bernoulli_rng": bernoulli_key}
    input_x = jnp.array([[1, 0, 15, 11]], dtype=jnp.float32)
    random_obfuscator = tabnet.RandomObfuscator()
    parameters = random_obfuscator.init(rngs=rngs_keys, input_x=input_x)
    actual_output = random_obfuscator.apply(
        variables={"params": parameters},
        input_x=input_x,
        rngs={"bernoulli_rng": bernoulli_key},
    )
    self.assertEqual(
        [actual_output[0].tolist(), actual_output[1].tolist()],
        [[[0, 0, 15, 11]], [[True, False, False, False]]],
    )


class GhostBatchNormalizationTest(absltest.TestCase):

  def test_ghost_batch_normalization(self):
    input_x = jnp.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=jnp.float32)
    ghost_batch_normalization = tabnet.GhostBatchNormalization(
        virtual_batch_size=1
    )
    parameters = ghost_batch_normalization.init(
        rngs={"params": jax.random.PRNGKey(0)}, input_x=input_x
    )
    actual_output = ghost_batch_normalization.apply(
        variables=parameters, input_x=input_x
    )
    expected_output = [
        [0.999995, 1.99999, 2.999985, 3.99998, 4.999975],
        [5.99997, 6.9999647, 7.99996, 8.999955, 9.99995],
    ]
    self.assertTrue(jnp.array_equal(actual_output.tolist(), expected_output))


class GatedLinearUnitLayerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="without_fully_connected_layer",
          fully_connected_layer=None,
          expected_output=[
              [-0.11091334, -0.15342952],
              [-0.00337686, -0.0011541],
          ],
      ),
      dict(
          testcase_name="with_fully_connected_layer",
          fully_connected_layer=nn.Dense(
              4,
              use_bias=False,
              kernel_init=_TEST_KERNEL_INIT,
              dtype=jnp.float32,
          ),
          expected_output=[[-2.0004694, -6.038208], [-6.145095, -15.874868]],
      ),
  )
  def test_gated_linear_unit_layer(
      self,
      fully_connected_layer,
      expected_output,
  ):
    input_x = jnp.array([[1, 2, 3, 4], [6, 7, 8, 9]], dtype=jnp.float32)
    gated_linear_unit_layer = tabnet.GatedLinearUnitLayer(
        input_dimensions=4,
        output_dimensions=2,
        fully_connected_layer=fully_connected_layer,
        virtual_batch_size=2,
    )
    parameters = gated_linear_unit_layer.init(
        rngs={"params": jax.random.PRNGKey(0)}, input_x=input_x
    )
    self.assertTrue(
        jnp.allclose(
            gated_linear_unit_layer.apply(
                variables=parameters, input_x=input_x
            ),
            jnp.array(expected_output, dtype=jnp.float32),
        )
    )


class GatedLinearUnitBlockTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="without_shared_layers",
          shared_layers=None,
          expected_output=[
              [10.540033, -5.5184975, 10.364549, 8.575027],
              [10.540033, -5.5184975, 10.364549, 8.575027],
          ],
      ),
      dict(
          testcase_name="with_shared_layers",
          shared_layers=(
              nn.Dense(
                  8,
                  use_bias=False,
                  kernel_init=_TEST_KERNEL_INIT,
                  dtype=jnp.float32,
              ),
              nn.Dense(
                  8,
                  use_bias=False,
                  kernel_init=_TEST_KERNEL_INIT,
                  dtype=jnp.float32,
              ),
          ),
          expected_output=[
              [-1.4125396e01, -2.4504197e-04, 1.4822230e01, 5.1411934e00],
              [-1.4125396e01, -2.4504197e-04, 1.4822230e01, 5.1411934e00],
          ],
      ),
      dict(
          testcase_name="with_first_attribute_enabled",
          shared_layers=None,
          expected_output=[
              [10.540033, -5.5184975, 10.364549, 8.575027],
              [10.540033, -5.5184975, 10.364549, 8.575027],
          ],
      ),
  )
  def test_gated_linear_unit_block(
      self,
      shared_layers,
      expected_output,
  ):
    input_x = jnp.array([[1, 0, 15, 11], [1, 0, 15, 11]], dtype=jnp.float32)
    gated_linear_unit = tabnet.GatedLinearUnitBlock(
        input_dimensions=4,
        output_dimensions=4,
        shared_layers=shared_layers,
        virtual_batch_size=2,
    )
    parameters = gated_linear_unit.init(
        rngs={"params": jax.random.PRNGKey(0)}, input_x=input_x
    )
    actual_output = gated_linear_unit.apply(
        variables=parameters, input_x=input_x
    )
    self.assertTrue(
        jnp.allclose(
            actual_output, jnp.array(expected_output, dtype=jnp.float32)
        )
    )


class IdentityTest(absltest.TestCase):

  def test_identity(self):
    input_x = jnp.array([[1, 2, 3, 4, 5]], dtype=jnp.float32)
    identify = tabnet.Identity()
    parameters = identify.init(
        rngs={"params": jax.random.PRNGKey(0)}, input_x=input_x
    )
    self.assertTrue(
        identify.apply(
            variables={"params": parameters}, input_x=input_x
        ).tolist(),
        [[1, 2, 3, 4, 5]],
    )


class FeatureTransformerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="with_identity",
          number_of_independent_glu_layers=0,
          shared_layers=None,
          expected_output=[
              [1, 0, 15, 11],
              [1, 0, 15, 11],
          ],
      ),
      dict(
          testcase_name="with_shared_layers",
          number_of_independent_glu_layers=1,
          shared_layers=(
              nn.Dense(
                  8,
                  use_bias=False,
                  kernel_init=_TEST_KERNEL_INIT,
                  dtype=jnp.float32,
              ),
              nn.Dense(
                  8,
                  use_bias=False,
                  kernel_init=_TEST_KERNEL_INIT,
                  dtype=jnp.float32,
              ),
          ),
          expected_output=[
              [-7.2668653, -1.7132494, -3.5207632, -8.867629],
              [-7.2668653, -1.7132494, -3.5207632, -8.867629],
          ],
      ),
  )
  def test_feature_transformer(
      self,
      number_of_independent_glu_layers,
      shared_layers,
      expected_output,
  ):
    input_x = jnp.array([[1, 0, 15, 11], [1, 0, 15, 11]], dtype=jnp.float32)
    feature_transformer = tabnet.FeatureTransformer(
        input_dimensions=4,
        output_dimensions=4,
        number_of_independent_glu_layers=number_of_independent_glu_layers,
        shared_layers=shared_layers,
        virtual_batch_size=2,
    )
    parameters = feature_transformer.init(
        rngs={"params": jax.random.PRNGKey(0)}, input_x=input_x
    )
    self.assertTrue(
        jnp.allclose(
            feature_transformer.apply(variables=parameters, input_x=input_x),
            jnp.array(expected_output, dtype=jnp.float32),
        )
    )


class SparseMaxTest(absltest.TestCase):

  def test_sparsemax(self):
    self.assertTrue(
        jnp.allclose(
            tabnet.sparsemax(input_x=jnp.array([[1, 2, 3]]), axis=0),
            jnp.array([1.0, 1.0, 1.0]),
        )
    )


class AttentiveTransformerTest(absltest.TestCase):

  def test_attentive_transformer_sparsemax_activation(self):
    input_x = jnp.array([[1, 2, 3, 4], [6, 7, 8, 9]], dtype=jnp.float32)
    priors = jnp.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=jnp.float32)
    layer = tabnet.AttentiveTransformer(
        input_dimensions=4,
        output_dimensions=4,
        virtual_batch_size=2,
    )
    parameters = layer.init(
        rngs={"params": jax.random.PRNGKey(0)}, priors=priors, input_x=input_x
    )
    self.assertTrue(
        jnp.array_equal(
            layer.apply(variables=parameters, priors=priors, input_x=input_x),
            jnp.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=jnp.float32),
        )
    )

  def test_attentive_transformer_not_implemented_error(self):
    input_x = jnp.array([[1, 2, 3, 4], [6, 7, 8, 9]], dtype=jnp.float32)
    priors = jnp.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=jnp.float32)
    with self.assertRaisesRegex(
        NotImplementedError, "sparsemax is implemented"
    ):
      attentive_transformer = tabnet.AttentiveTransformer(
          input_dimensions=4,
          output_dimensions=4,
          virtual_batch_size=2,
          mask_type="entmax",
      )
      parameters = attentive_transformer.init(
          rngs={"params": jax.random.PRNGKey(0)}, priors=priors, input_x=input_x
      )
      attentive_transformer.apply(
          variables={"params": parameters}, priors=priors, input_x=input_x
      )


class TabNetEncoderTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="with_shared_glu_layers",
          shared_glu_layers=2,
          expected_output=[
              [
                  0.0,
                  0.0,
                  0.30581588,
                  0.05495514,
                  0.0,
                  0.35570237,
                  0.0714035,
                  0.3608738,
              ],
              [
                  0.25762382,
                  0.3658545,
                  0.0,
                  0.0,
                  0.0,
                  1.2117158,
                  0.12706628,
                  0.75093335,
              ],
          ],
      ),
      dict(
          testcase_name="without_shared_glu_layers",
          shared_glu_layers=0,
          expected_output=[
              [
                  0.0,
                  0.46003976,
                  0.0,
                  0.6475601,
                  0.0,
                  0.0,
                  1.1979795,
                  0.64231074,
              ],
              [
                  0.17106369,
                  1.6240108,
                  0.0,
                  0.36196068,
                  0.0,
                  0.0,
                  3.7052367,
                  1.8403097,
              ],
          ],
      ),
  )
  def test_tabnet_encoder(
      self,
      shared_glu_layers,
      expected_output,
  ):
    input_x = jnp.array([[1, 2, 3, 4], [6, 7, 8, 9]], dtype=jnp.float32)
    prior = jnp.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=jnp.float32)
    tabnet_encoder = tabnet.TabNetEncoder(
        input_dimensions=4,
        output_dimensions=4,
        shared_glu_layers=shared_glu_layers,
        virtual_batch_size=2,
    )
    parameters = tabnet_encoder.init(
        rngs={"params": jax.random.PRNGKey(0)},
        input_x=input_x,
        prior=prior,
        forward_masks=False,
    )
    attentive_transformer_explain, _ = tabnet_encoder.apply(
        variables=parameters,
        input_x=input_x,
        prior=prior,
        forward_masks=False,
    )
    self.assertTrue(
        jnp.allclose(
            attentive_transformer_explain[0], jnp.asarray(expected_output)
        )
    )

  def test_tabnet_encoder_without_prior(self):
    input_x = jnp.array([[1, 2, 3, 4], [6, 7, 8, 9]], dtype=jnp.float32)
    tabnet_encoder = tabnet.TabNetEncoder(
        input_dimensions=4,
        output_dimensions=4,
        virtual_batch_size=2,
    )
    parameters = tabnet_encoder.init(
        rngs={"params": jax.random.PRNGKey(0)},
        input_x=input_x,
        prior=None,
        forward_masks=False,
    )
    actual_output = tabnet_encoder.apply(
        variables=parameters,
        input_x=input_x,
        prior=None,
        forward_masks=False,
    )
    expected_value = jnp.array(
        [
            [
                0.0,
                0.0,
                0.30581588,
                0.05495514,
                0.0,
                0.35570237,
                0.0714035,
                0.3608738,
            ],
            [
                0.25762382,
                0.3658545,
                0.0,
                0.0,
                0.0,
                1.2117158,
                0.12706628,
                0.75093335,
            ],
        ],
        dtype=jnp.float32,
    )
    self.assertTrue(jnp.allclose(actual_output[0][0], expected_value))

  def test_tabnet_encoder_forward_masks(self):
    input_x = jnp.array([[1, 2, 3, 4], [6, 7, 8, 9]], dtype=jnp.float32)
    prior = jnp.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=jnp.float32)
    tabnet_encoder = tabnet.TabNetEncoder(
        input_dimensions=4,
        output_dimensions=4,
        shared_glu_layers=0,
        virtual_batch_size=2,
    )
    parameters = tabnet_encoder.init(
        rngs={"params": jax.random.PRNGKey(0)},
        input_x=input_x,
        prior=prior,
        forward_masks=True,
    )
    actual_output = tabnet_encoder.apply(
        variables=parameters,
        input_x=input_x,
        prior=prior,
        forward_masks=True,
    )
    self.assertTrue(
        jnp.allclose(
            actual_output[0],
            jnp.array(
                [
                    [0.04559661, 2.125475, 0.0, 3.6321173],
                    [0.9563913, 4.811488, 18.833267, 7.7025824],
                ],
                dtype=jnp.float32,
            ),
        )
    )


class TabNetDecoderTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="with_shared_glu_layers",
          shared_glu_layers=1,
          expected=[[3.1984415, -1.8801607], [3.1984415, -1.8801607]],
      ),
      dict(
          testcase_name="without_shared_glu_layers",
          shared_glu_layers=0,
          expected=[[-3.1566164, -1.1233839], [-3.1566164, -1.1233839]],
      ),
  )
  def test_tabnet_decoder(
      self,
      shared_glu_layers,
      expected,
  ):
    input_x = jnp.array(
        [[[1, 2, 3, 4, 6, 7, 8, 9], [1, 2, 3, 4, 6, 7, 8, 9]]],
        dtype=jnp.float32,
    )
    tabnet_decoder = tabnet.TabNetDecoder(
        input_dimensions=2,
        shared_glu_layers=shared_glu_layers,
        virtual_batch_size=1,
    )
    parameters = tabnet_decoder.init(
        rngs={"params": jax.random.PRNGKey(0)}, input_x=input_x
    )
    self.assertTrue(
        jnp.allclose(
            tabnet_decoder.apply(variables=parameters, input_x=input_x),
            jnp.array(expected, dtype=jnp.float32),
        )
    )


class TabNetPretrainingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="successive_network_steps_value_error",
          successive_network_steps=0,
          independent_glu_layers=2,
          shared_glu_layers=2,
          error_message="successive_network_steps should",
      ),
      dict(
          testcase_name="independent_glu_layers_shared_glu_layers_value_error",
          successive_network_steps=3,
          independent_glu_layers=0,
          shared_glu_layers=0,
          error_message="shared_glu_layers and independent_glu_layers",
      ),
  )
  def test_tabnet_pretraining_value_error(
      self,
      successive_network_steps,
      independent_glu_layers,
      shared_glu_layers,
      error_message,
  ):
    rng_key, bernoulli_key = jax.random.split(jax.random.PRNGKey(0))
    rng_keys = {"params": rng_key, "bernoulli_rng": bernoulli_key}
    input_x = jnp.array([[[1, 2], [1, 2]]], dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, error_message):
      tabnet_pretraining = tabnet.TabNetPretraining(
          input_dimensions=2,
          successive_network_steps=successive_network_steps,
          independent_glu_layers=independent_glu_layers,
          shared_glu_layers=shared_glu_layers,
          virtual_batch_size=2,
      )
      parameters = tabnet_pretraining.init(rngs=rng_keys, input_x=input_x)
      tabnet_pretraining.apply(
          variables={"params": parameters},
          input_x=input_x,
          rngs={"bernoulli_rng": bernoulli_key},
      )

  def test_tabnet_pretraining_not_train(self):
    rng_key, bernoulli_key = jax.random.split(jax.random.PRNGKey(0))
    rng_keys = {"params": rng_key, "bernoulli_rng": bernoulli_key}
    input_x = jnp.array([[1, 2], [1, 2]], dtype=jnp.float32)
    tabnet_pretraining = tabnet.TabNetPretraining(
        input_dimensions=2,
        virtual_batch_size=2,
        categorical_embedding_dimensions=None,
    )
    parameters = tabnet_pretraining.init(rngs=rng_keys, input_x=input_x)
    actual_output = tabnet_pretraining.apply(
        variables=parameters,
        input_x=input_x,
        rngs={"bernoulli_rng": bernoulli_key},
    )
    self.assertTrue(
        jnp.allclose(
            actual_output[0],
            jnp.array(
                [[0.06562408, 0.1289592], [0.06562408, 0.1289592]],
                dtype=jnp.float32,
            ),
        )
    )

  def test_tabnet_pretraining_train(self):
    rng_key, bernoulli_key = jax.random.split(jax.random.PRNGKey(0))
    rng_keys = {"params": rng_key, "bernoulli_rng": bernoulli_key}
    input_x = jnp.array([[1, 2], [1, 2]], dtype=jnp.float32)
    tabnet_pretraining = tabnet.TabNetPretraining(
        input_dimensions=2,
        virtual_batch_size=2,
        categorical_embedding_dimensions=None,
    )
    parameters = tabnet_pretraining.init(rngs=rng_keys, input_x=input_x)
    actual_output, _ = tabnet_pretraining.apply(
        variables=parameters,
        input_x=input_x,
        train=True,
        mutable=["batch_stats"],
        rngs={"bernoulli_rng": bernoulli_key},
    )
    self.assertTrue(
        jnp.allclose(
            actual_output[0],
            jnp.array(
                [[0.0, 0.0], [0.0, 0.0]],
                dtype=jnp.float32,
            ),
            atol=1e-05,
        )
    )


class TabNetCoreTrainingTest(absltest.TestCase):

  def test_tabnet_core_training_not_multi_task(self):
    input_x = jnp.array([[1, 2], [1, 2]], dtype=jnp.float32)
    tabnet_core_training = tabnet.TabNetCoreTraining(
        input_dimensions=2,
        output_dimensions=2,
        virtual_batch_size=2,
    )
    parameters = tabnet_core_training.init(
        rngs={"params": jax.random.PRNGKey(0)}, input_x=input_x
    )
    actual_output = tabnet_core_training.apply(
        variables=parameters, input_x=input_x
    )
    self.assertTrue(
        jnp.allclose(
            actual_output[0],
            jnp.array(
                [[-1.0982076, -0.4056675], [-1.0982076, -0.4056675]],
                dtype=jnp.float32,
            ),
        )
    )

  def test_tabnet_core_training_multi_task(self):
    input_x = jnp.array([[1, 2], [1, 2]], dtype=jnp.float32)
    tabnet_core_training = tabnet.TabNetCoreTraining(
        input_dimensions=2,
        output_dimensions=(2, 2),
        virtual_batch_size=2,
    )
    parameters = tabnet_core_training.init(
        rngs={"params": jax.random.PRNGKey(0)}, input_x=input_x
    )
    actual_output = tabnet_core_training.apply(
        variables=parameters, input_x=input_x
    )
    self.assertTrue(
        jnp.allclose(
            jnp.array(actual_output[0][0], dtype=jnp.float32),
            jnp.array(
                [[-0.5915043, -0.80630255], [-0.5915043, -0.80630255]],
                dtype=jnp.float32,
            ),
        )
    )

  def test_tabnet_core_training_forward_masks(self):
    input_x = jnp.array([[1, 2], [1, 2]], dtype=jnp.float32)
    tabnet_core_training = tabnet.TabNetCoreTraining(
        input_dimensions=2,
        output_dimensions=2,
        virtual_batch_size=2,
    )
    parameters = tabnet_core_training.init(
        rngs={"params": jax.random.PRNGKey(0)},
        input_x=input_x,
        forward_masks=True,
    )
    actual_output = tabnet_core_training.apply(
        variables=parameters, input_x=input_x, forward_masks=True
    )
    self.assertTrue(
        jnp.allclose(
            actual_output[0],
            jnp.array(
                [
                    [0.98145294, 0.96103907],
                    [0.98145294, 0.96103907],
                ],
                dtype=jnp.float32,
            ),
        )
    )


class TabNetTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="successive_network_steps_value_error",
          successive_network_steps=0,
          independent_glu_layers=2,
          shared_glu_layers=2,
          error_message="successive_network_steps should",
      ),
      dict(
          testcase_name="independent_glu_layers_shared_glu_layers_value_error",
          successive_network_steps=3,
          independent_glu_layers=0,
          shared_glu_layers=0,
          error_message="shared_glu_layers and independent_glu_layers can not",
      ),
  )
  def test_tabnet_value_error(
      self,
      successive_network_steps,
      independent_glu_layers,
      shared_glu_layers,
      error_message,
  ):
    input_x = jnp.array([[[1, 2], [1, 2]]], dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, error_message):
      tabnet_model = tabnet.TabNet(
          input_dimensions=2,
          output_dimensions=2,
          successive_network_steps=successive_network_steps,
          independent_glu_layers=independent_glu_layers,
          shared_glu_layers=shared_glu_layers,
          virtual_batch_size=2,
      )
      parameters = tabnet_model.init(
          rngs={"params": jax.random.PRNGKey(0)}, input_x=input_x
      )
      tabnet_model.apply(
          variables={"params": parameters},
          input_x=input_x,
      )

  def test_tabnet_forward_masks(self):
    rng_key, _ = jax.random.split(jax.random.PRNGKey(0))
    input_x = jnp.array([[1, 2], [1, 2]], dtype=jnp.float32)
    tabnet_model = tabnet.TabNet(
        input_dimensions=2,
        output_dimensions=2,
        virtual_batch_size=2,
    )
    parameters = tabnet_model.init(
        rngs={"params": rng_key}, input_x=input_x, forward_masks=True
    )
    actual_output = tabnet_model.apply(
        variables=parameters, input_x=input_x, forward_masks=True
    )
    self.assertTrue(
        jnp.allclose(
            actual_output[0],
            jnp.array(
                [
                    [0.55870825, 0.4871775],
                    [0.55870825, 0.4871775],
                ],
                dtype=jnp.float32,
            ),
        )
    )

  def test_tabnet_not_train(self):
    input_x = jnp.array([[1, 2], [1, 2]], dtype=jnp.float32)
    tabnet_model = tabnet.TabNet(
        input_dimensions=2,
        output_dimensions=2,
        virtual_batch_size=2,
    )
    parameters = tabnet_model.init(
        rngs={"params": jax.random.PRNGKey(0)}, input_x=input_x
    )
    actual_output = tabnet_model.apply(
        variables=parameters,
        input_x=input_x,
    )
    self.assertTrue(
        jnp.allclose(
            actual_output[0],
            jnp.array(
                [[-1.420868, -0.2764181], [-1.420868, -0.2764181]],
                dtype=jnp.float32,
            ),
        )
    )


class TabNetAgentTests(absltest.TestCase):

  def test_tabnet_agent(self):
    self.assertIsInstance(
        tabnet.TabNetAgent(hyperparameters=_TEST_TABNET_HYPERPARAMETERS),
        tabnet.TabNetAgent,
    )

  def test_tabnet_agent_build_flax_module(self):
    tabnet_flax_module = tabnet.TabNetAgent(
        hyperparameters=_TEST_TABNET_HYPERPARAMETERS
    ).build_flax_module()
    self.assertIsInstance(tabnet_flax_module, tabnet.TabNet)

  def test_tabnet_agent_get_dummy_input(self):
    self.assertEqual(
        tabnet.TabNetAgent(hyperparameters=_TEST_TABNET_HYPERPARAMETERS)
        .get_dummy_inputs()
        .tolist(),
        [[0.0, 0.0, 0.0, 0.0]],
    )

  def test_tabnet_agent_predict(self):
    tabnet_agent = tabnet.TabNetAgent(
        hyperparameters=_TEST_TABNET_HYPERPARAMETERS
    )
    tabnet_flax_module = tabnet_agent.build_flax_module()
    test_dummy_init_batch = tabnet_agent.get_dummy_inputs()
    parameters = jax.jit(tabnet_flax_module.init, static_argnames="train")(
        rngs={"params": jax.random.PRNGKey(0)},
        input_x=test_dummy_init_batch,
        train=False,
    )
    model_state = base_agent.BaseAgentState.create(
        apply_fn=tabnet_flax_module.apply,
        params=parameters["params"],
        tx=optax.adam(0.1),
        batch_stats=parameters["batch_stats"],
        exploration_exploitation_epsilon=0.0,
    )
    actual_outcome = tabnet_agent.predict(
        agent_state=model_state,
        batch=jnp.array([[1, 2, 3, 4]], dtype=jnp.float32),
        prediction_seed=0,
    )
    self.assertEqual(actual_outcome.action.item(), 0)

if __name__ == "__main__":
  absltest.main()
