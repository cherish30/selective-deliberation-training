# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# setup.py is the fallback installation script when pyproject.toml does not work
import os
from pathlib import Path

from setuptools import find_packages, setup

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "verl/version/version")) as f:
    __version__ = f.read().strip()

install_requires = [
    "torch==2.6.0",
    "transformers==4.51.1",
    "peft==0.19.1",
    "accelerate==1.13.0",
    "ray[default]==2.55.1",
    "tensordict==0.6.2",
    "torchdata==0.11.0",
    "datasets==4.8.5",
    "pyarrow==24.0.0",
    "dill==0.4.1",
    "numpy",
    "pandas",
    "wandb==0.27.0",
    "hydra-core==1.3.2",
    "packaging==26.2",
    "uvicorn==0.46.0",
    "fastapi==0.136.1",
    "pybind11==3.0.4",
    "pylatexenc==2.10",
    "codetiming==1.4.0",
    "qwen-vl-utils==0.0.14",
    "alfworld==0.4.2",
    "smolagents==1.25.0",
]

TEST_REQUIRES = ["pytest", "py-spy"]
GPU_REQUIRES = ["liger-kernel==0.8.0", "flash-attn==2.7.4.post1"]
VLLM_REQUIRES = ["vllm==0.8.5"]

extras_require = {
    "test": TEST_REQUIRES,
    "gpu": GPU_REQUIRES,
    "vllm": VLLM_REQUIRES,
}


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="verl",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where="."),
    url="https://github.com/volcengine/verl",
    license="Apache 2.0",
    author="Anonymous",
    author_email="",
    description="Selective Deep Thinking for Language Agents (built on verl)",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        "": ["version/*"],
        "verl": ["trainer/config/*.yaml"],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
