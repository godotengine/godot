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
      'target_name': 'crashpad_test',
      'type': 'static_library',
      'dependencies': [
        '../compat/compat.gyp:crashpad_compat',
        '../third_party/gtest/gtest.gyp:gtest',
        '../third_party/mini_chromium/mini_chromium.gyp:base',
        '../util/util.gyp:crashpad_util',
      ],
      'include_dirs': [
        '..',
      ],
      'sources': [
        'errors.cc',
        'errors.h',
        'file.cc',
        'file.h',
        'filesystem.cc',
        'filesystem.h',
        'gtest_death.h',
        'gtest_disabled.cc',
        'gtest_disabled.h',
        'hex_string.cc',
        'hex_string.h',
        'linux/fake_ptrace_connection.cc',
        'linux/fake_ptrace_connection.h',
        'linux/get_tls.cc',
        'linux/get_tls.h',
        'mac/dyld.cc',
        'mac/dyld.h',
        'mac/exception_swallower.cc',
        'mac/exception_swallower.h',
        'mac/mach_errors.cc',
        'mac/mach_errors.h',
        'mac/mach_multiprocess.cc',
        'mac/mach_multiprocess.h',
        'main_arguments.cc',
        'main_arguments.h',
        'multiprocess.h',
        'multiprocess_exec.cc',
        'multiprocess_exec.h',
        'multiprocess_exec_posix.cc',
        'multiprocess_exec_win.cc',
        'multiprocess_posix.cc',
        'process_type.cc',
        'process_type.h',
        'scoped_module_handle.cc',
        'scoped_module_handle.h',
        'scoped_temp_dir.cc',
        'scoped_temp_dir.h',
        'scoped_temp_dir_posix.cc',
        'scoped_temp_dir_win.cc',
        'test_paths.cc',
        'test_paths.h',
        'win/child_launcher.cc',
        'win/child_launcher.h',
        'win/win_child_process.cc',
        'win/win_child_process.h',
        'win/win_multiprocess.cc',
        'win/win_multiprocess.h',
        'win/win_multiprocess_with_temp_dir.cc',
        'win/win_multiprocess_with_temp_dir.h',
      ],
      'direct_dependent_settings': {
        'include_dirs': [
          '..',
        ],
      },
      'conditions': [
        ['OS=="mac"', {
          'dependencies': [
            '../handler/handler.gyp:crashpad_handler_lib',
            '../snapshot/snapshot.gyp:crashpad_snapshot',
          ],
          'link_settings': {
            'libraries': [
              '$(SDKROOT)/usr/lib/libbsm.dylib',
            ],
          },
        }],
        ['OS=="win"', {
          'link_settings': {
            'libraries': [
              '-lshell32.lib',
            ],
          },
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
      'target_name': 'crashpad_gmock_main',
      'type': 'static_library',
      'dependencies': [
        'crashpad_test',
        '../third_party/gtest/gmock.gyp:gmock',
        '../third_party/gtest/gtest.gyp:gtest',
        '../third_party/mini_chromium/mini_chromium.gyp:base',
      ],
      'include_dirs': [
        '..',
      ],
      'defines': [
        'CRASHPAD_TEST_LAUNCHER_GMOCK=1',
      ],
      'sources': [
        'gtest_main.cc',
      ],
    },
    {
      'target_name': 'crashpad_gtest_main',
      'type': 'static_library',
      'dependencies': [
        'crashpad_test',
        '../third_party/gtest/gtest.gyp:gtest',
        '../third_party/mini_chromium/mini_chromium.gyp:base',
      ],
      'include_dirs': [
        '..',
      ],
      'defines': [
        'CRASHPAD_TEST_LAUNCHER_GTEST=1',
      ],
      'sources': [
        'gtest_main.cc',
      ],
    },
  ],
}
