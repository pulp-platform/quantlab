# 
# __init__.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

from .statistics.statistics            import *                      # parent classes for all the other statistics
from .statistics.profilingstatistic    import ProfilingStatistic     # general-purpose statistic
from .statistics.lossstatistic         import LossStatistic          # general-purpose statistic
from .statistics.taskstatistic         import TaskStatistic          # template for problem-specific metrics
from .statistics.learningratestatistic import LearningRateStatistic  # general-purpose statistic
from .statistics.tensorstatistics      import *                      # general-purpose statistics
from .statistics.visionstatistics      import RGBInputsSnapshot      # template for problem-specific input snapshooting
from .statistics.visionstatistics      import *                      # general-purpose statistics for computer vision networks
from .writerstub                       import WriterStub
from .meter                            import Meter

