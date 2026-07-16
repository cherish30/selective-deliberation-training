# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# and the verl-agent (GiGPO) team.
#     http://www.apache.org/licenses/LICENSE-2.0
# and the verl-agent (GiGPO) team.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .alfworld import *

# Only the ALFWorld prompt module ships in this release (the paper trains and
# evaluates the RL policy on ALFWorld only). The other verl-agent (GiGPO)
# environment prompt modules are optional and not included; skip them if absent
# rather than failing import for users who only need ALFWorld.
try:
    from .webshop import *
except ModuleNotFoundError:
    pass
try:
    from .sokoban import *
except ModuleNotFoundError:
    pass
try:
    from .gym_cards import *
except ModuleNotFoundError:
    pass
try:
    from .appworld import *
except ModuleNotFoundError:
    pass
try:
    from .search import *
except ModuleNotFoundError:
    pass