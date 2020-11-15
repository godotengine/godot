# Copyright 2017 The Crashpad Authors. All rights reserved.
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
      'target_name': 'crashpad_handler_test',
      'type': 'executable',
      'dependencies': [
        'crashpad_handler_test_extended_handler',
        'handler.gyp:crashpad_handler_lib',
        '../client/client.gyp:crashpad_client',
        '../compat/compat.gyp:crashpad_compat',
        '../snapshot/snapshot.gyp:crashpad_snapshot',
        '../snapshot/snapshot_test.gyp:crashpad_snapshot_test_lib',
        '../test/test.gyp:crashpad_gtest_main',
        '../test/test.gyp:crashpad_test',
        '../third_party/gtest/gtest.gyp:gtest',
        '../third_party/mini_chromium/mini_chromium.gyp:base',
        '../util/util.gyp:crashpad_util',
      ],
      'include_dirs': [
        '..',
      ],
      'sources': [
        'crashpad_handler_test.cc',
        'linux/exception_handler_server_test.cc',
        'minidump_to_upload_parameters_test.cc',
      ],
      'conditions': [
        ['OS!="win"', {
          'dependencies!': [
            'crashpad_handler_test_extended_handler',
          ],
          'sources!': [
            'crashpad_handler_test.cc',
          ],
        }],
      ],
      'target_conditions': [
        ['OS=="android"', {
          'sources/': [
            ['include', '^linux/'],
          ],
        }],
      ],
    },
    {
      'target_name': 'crashpad_handler_test_extended_handler',
      'type': 'executable',
      'dependencies': [
        '../compat/compat.gyp:crashpad_compat',
        '../minidump/minidump_test.gyp:crashpad_minidump_test_lib',
        '../third_party/mini_chromium/mini_chromium.gyp:base',
        '../tools/tools.gyp:crashpad_tool_support',
        'handler.gyp:crashpad_handler_lib',
      ],
      'include_dirs': [
        '..',
      ],
      'sources': [
        'crashpad_handler_test_extended_handler.cc',
      ],
    },
  ],
  'conditions': [
    ['OS=="win"', {
      'targets': [
        {
          'target_name': 'crash_other_program',
          'type': 'executable',
          'dependencies': [
            '../client/client.gyp:crashpad_client',
            '../test/test.gyp:crashpad_test',
            '../third_party/gtest/gtest.gyp:gtest',
            '../third_party/mini_chromium/mini_chromium.gyp:base',
            '../util/util.gyp:crashpad_util',
          ],
          'sources': [
            'win/crash_other_program.cc',
          ],
        },
        {
          'target_name': 'crashy_program',
          'type': 'executable',
          'dependencies': [
            '../client/client.gyp:crashpad_client',
            '../third_party/mini_chromium/mini_chromium.gyp:base',
            '../util/util.gyp:crashpad_util',
          ],
          'include_dirs': [
            '..',
          ],
          'sources': [
            'win/crashy_test_program.cc',
          ],
        },
        {
          'target_name': 'crashy_signal',
          'type': 'executable',
          'dependencies': [
            '../client/client.gyp:crashpad_client',
            '../third_party/mini_chromium/mini_chromium.gyp:base',
          ],
          'include_dirs': [
            '..',
          ],
          'sources': [
            'win/crashy_signal.cc',
          ],
        },
        {
          'target_name': 'fake_handler_that_crashes_at_startup',
          'type': 'executable',
          'sources': [
            'win/fake_handler_that_crashes_at_startup.cc',
          ],
        },
        {
          'target_name': 'hanging_program',
          'type': 'executable',
          'dependencies': [
            '../client/client.gyp:crashpad_client',
            '../third_party/mini_chromium/mini_chromium.gyp:base',
          ],
          'sources': [
            'win/hanging_program.cc',
          ],
        },
        {
          'target_name': 'loader_lock_dll',
          'type': 'loadable_module',
          'sources': [
            'win/loader_lock_dll.cc',
          ],
          'msvs_settings': {
            'NoImportLibrary': 'true',
          },
        },
        {
          'target_name': 'self_destroying_program',
          'type': 'executable',
          'dependencies': [
            '../client/client.gyp:crashpad_client',
            '../compat/compat.gyp:crashpad_compat',
            '../snapshot/snapshot.gyp:crashpad_snapshot',
            '../third_party/mini_chromium/mini_chromium.gyp:base',
            '../util/util.gyp:crashpad_util',
          ],
          'include_dirs': [
            '..',
          ],
          'sources': [
            'win/self_destroying_test_program.cc',
          ],
        },
      ],
      'conditions': [
        # Cannot create an x64 DLL with embedded debug info.
        ['target_arch=="ia32"', {
          'targets': [
            {
              'target_name': 'crashy_z7_loader',
              'type': 'executable',
              'dependencies': [
                '../client/client.gyp:crashpad_client',
                '../test/test.gyp:crashpad_test',
                '../third_party/mini_chromium/mini_chromium.gyp:base',
              ],
              'include_dirs': [
                '..',
              ],
              'sources': [
                'win/crashy_test_z7_loader.cc',
              ],
            },
          ],
        }],
      ],
    }],
  ],
}
