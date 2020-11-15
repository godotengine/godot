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
    '../build/crashpad.gypi',
  ],
  'targets': [
    {
      'target_name': 'crashpad_test_test',
      'type': 'executable',
      'dependencies': [
        'crashpad_test_test_multiprocess_exec_test_child',
        'test.gyp:crashpad_gmock_main',
        'test.gyp:crashpad_test',
        '../compat/compat.gyp:crashpad_compat',
        '../third_party/gtest/gmock.gyp:gmock',
        '../third_party/gtest/gtest.gyp:gtest',
        '../third_party/mini_chromium/mini_chromium.gyp:base',
        '../util/util.gyp:crashpad_util',
      ],
      'include_dirs': [
        '..',
      ],
      'sources': [
        'hex_string_test.cc',
        'mac/mach_multiprocess_test.cc',
        'main_arguments_test.cc',
        'multiprocess_exec_test.cc',
        'multiprocess_posix_test.cc',
        'scoped_temp_dir_test.cc',
        'test_paths_test.cc',
        'win/win_child_process_test.cc',
        'win/win_multiprocess_test.cc',
      ],
    },
    {
      'target_name': 'crashpad_test_test_multiprocess_exec_test_child',
      'type': 'executable',
      'dependencies': [
        '../third_party/mini_chromium/mini_chromium.gyp:base',
      ],
      'sources': [
        'multiprocess_exec_test_child.cc',
      ],
    },
  ],
}
