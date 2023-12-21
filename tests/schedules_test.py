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

"""Tests for utility functions in schedules.py."""
from absl.testing import absltest
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
from optimus.trainer_lib import schedules


class SchedulesTests(absltest.TestCase):

  def test_check_schedule_hyperparameters(self):
    schedule_hyperparameters = config_dict.ConfigDict(
        dict(schedule="test_schedule", value_one=1, value_two=2)
    )
    self.assertIsNone(
        schedules.check_schedule_hyperparameters(
            schedule_hyperparameters=schedule_hyperparameters,
            expected_keys=(
                "value_one",
                "value_two",
            ),
        )
    )

  def test_check_schedule_hyperparameters_value_error(self):
    schedule_hyperparameters = config_dict.ConfigDict(
        dict(value_one=1, value_two=2)
    )
    with self.assertRaisesRegex(
        ValueError, "schedule_hyperparameters keys are"
    ):
      schedules.check_schedule_hyperparameters(
          schedule_hyperparameters=schedule_hyperparameters,
          expected_keys=(
              "value_one",
              "value_two",
              "value_three",
          ),
      )

  def test_linear_schedule(self):
    schedule_hyperparameters = config_dict.ConfigDict(
        dict(
            schedule="linear_schedule",
            initial_value=0.1,
            end_value=0.0,
            transition_steps=0,
        )
    )
    self.assertEqual(
        schedules.linear_schedule(
            schedule_hyperparameters=schedule_hyperparameters
        )(1),
        0.1,
    )

  def test_constant_schedule(self):
    schedule_hyperparameters = config_dict.ConfigDict(
        dict(
            schedule="constant_schedule",
            initial_value=0.1,
        )
    )
    self.assertEqual(
        schedules.constant_schedule(
            schedule_hyperparameters=schedule_hyperparameters
        )(10),
        0.1,
    )

  def test_warmup_exponential_decay_schedule(self):
    schedule_hyperparameters = config_dict.ConfigDict(
        dict(
            schedule="warmup_exponential_decay_schedule",
            initial_value=0.0,
            peak_value=0.1,
            warmup_steps=10,
            transition_steps=20,
            decay_rate=0.0001,
            transition_begin=0,
            staircase=False,
            end_value=None,
        )
    )
    self.assertTrue(
        jnp.allclose(
            schedules.warmup_exponential_decay_schedule(
                schedule_hyperparameters=schedule_hyperparameters
            )(19),
            jnp.asarray(0.00158489),
        )
    )

  def test_get_schedule(self):
    self.assertEqual(
        schedules.get_schedule(schedule_name="linear_schedule"),
        schedules.linear_schedule,
    )

  def test_get_schedule_lookup_error(self):
    with self.assertRaisesRegex(LookupError, "Unrecognized schedule name"):
      schedules.get_schedule(schedule_name="other_schedule")


if __name__ == "__main__":
  absltest.main()
