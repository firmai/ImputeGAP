# coding=utf-8
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

'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings

import numpy as np

from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils
from imputegap.wrapper.AlgoPython.GAIN.gain import gain

warnings.simplefilter(action='ignore', category=FutureWarning)



def gainRecovery (miss_data_x, batch_size=32, hint_rate=0.9, alpha=10, epoch=100):
  '''Main function for UCI letter and spam datasets.
  
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''

  input_data = np.copy(miss_data_x)

  print("\t\t(PYTHON) GAIN: Matrix Shape: (", input_data.shape[0], ", ", input_data.shape[1], ") "
        "for batch_size ", batch_size, ", hint_rate ", hint_rate, ", alpha ", alpha, ", and epoch ", epoch, "...")
  
  gain_parameters = {'batch_size': batch_size,
                     'hint_rate': hint_rate,
                     'alpha': alpha,
                     'iterations': epoch}

  if batch_size == -1:
      batch_size = input_data.shape[1]//1
  gain_parameters['batch_size'] = batch_size
  # print('Batch size: ', gain_parameters["batch_size"])

  max_trials = 3  # Maximum number of retries
  trial = 0

  while trial < max_trials:
      imputed_data_x = gain(input_data, gain_parameters)

      if not np.all(np.isnan(imputed_data_x)):
          return imputed_data_x  # Successfully imputed, return result

      print(f"\t\t(PYTHON) GAIN: Trial {trial + 1} failed, reattempting...")
      trial += 1

  print("\t\t(PYTHON) GAIN: All trials failed, returning last imputed result.")
  return imputed_data_x  # Return last attempt even if NaN
