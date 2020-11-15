# Copyright 2014 The Crashpad Authors. All rights reserved.
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
      'target_name': 'crashpad_client_test',
      'type': 'executable',
      'dependencies': [
        'client.gyp:crashpad_client',
        '../compat/compat.gyp:crashpad_compat',
        '../handler/handler.gyp:crashpad_handler',
        '../snapshot/snapshot.gyp:crashpad_snapshot',
        '../test/test.gyp:crashpad_gmock_main',
        '../test/test.gyp:crashpad_test',
        '../third_party/gtest/gmock.gyp:gmock',
        '../third_party/gtest/gtest.gyp:gtest',
        '../third_party/mini_chromium/mini_chromium.gyp:base',
        '../util/util.gyp:crashpad_util',
      ],
      'include_dirs': [
        '..',
      ],
      'sources': [
        'annotation_test.cc',
        'annotation_list_test.cc',
        'crash_report_database_test.cc',
        'crashpad_client_win_test.cc',
        'crashpad_client_linux_test.cc',
        'prune_crash_reports_test.cc',
        'settings_test.cc',
        'simple_address_range_bag_test.cc',
        'simple_string_dictionary_test.cc',
        'simulate_crash_mac_test.cc',
      ],
      'conditions': [
        ['OS=="win"', {
          'dependencies': [
            '../handler/handler.gyp:crashpad_handler_console',
          ],
        }],
      ],
      'target_conditions': [
        ['OS=="android"', {
          'sources/': [
            ['include', '^crashpad_client_linux_test\\.cc$'],
          ],
        }],
      ],
    },
  ],
}
