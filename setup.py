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

"""The setup script for Optimus."""
import pathlib
import setuptools

with pathlib.Path("requirements.txt").open() as requirements_path:
  requirements = requirements_path.read().splitlines()

setuptools.setup(
    name="optimus",
    version="0.0.1",
    author="Google gTech Ads EMEA Privacy Data Science Team",
    license="MIT",
    packages=setuptools.find_packages(
        include=[
            "optimus",
            "optimus.*",
        ]
    ),
    install_requires=requirements,
    extras_require={},
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="1pd privacy reinforcement learning optimization personalization",
)
