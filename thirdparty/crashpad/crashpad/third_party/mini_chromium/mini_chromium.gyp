# Copyright 2015 The Crashpad Authors. All rights reserved.
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

{
  'includes': [
    '../../build/crashpad_dependencies.gypi',
  ],
  'targets': [
    {
      # To support Crashpad’s standalone build and its build depending on
      # external libraries, Crashpad code depending on base should do so through
      # this shim, which will either get base from mini_chromium or an external
      # library depending on the build type.
      'target_name': 'base',
      'type': 'none',
      'conditions': [
        ['crashpad_dependencies=="standalone"', {
          'dependencies': [
            'mini_chromium/base/base.gyp:base',
          ],
          'export_dependent_settings': [
            'mini_chromium/base/base.gyp:base',
          ],
        }],
        ['crashpad_dependencies=="external"', {
          'dependencies': [
            '../../../../mini_chromium/mini_chromium/base/base.gyp:base',
          ],
          'export_dependent_settings': [
            '../../../../mini_chromium/mini_chromium/base/base.gyp:base',
          ],
        }],
      ],
    },
  ],
}
