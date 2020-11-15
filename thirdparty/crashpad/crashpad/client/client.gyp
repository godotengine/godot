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
      'target_name': 'crashpad_client',
      'type': 'static_library',
      'dependencies': [
        '../compat/compat.gyp:crashpad_compat',
        '../third_party/mini_chromium/mini_chromium.gyp:base',
        '../util/util.gyp:crashpad_util',
      ],
      'include_dirs': [
        '..',
      ],
      'sources': [
        'annotation.cc',
        'annotation.h',
        'annotation_list.cc',
        'annotation_list.h',
        'crash_report_database.cc',
        'crash_report_database.h',
        'crash_report_database_mac.mm',
        'crash_report_database_win.cc',
        'crashpad_client.h',
        'crashpad_client_linux.cc',
        'crashpad_client_mac.cc',
        'crashpad_client_win.cc',
        'crashpad_info.cc',
        'crashpad_info.h',
        'prune_crash_reports.cc',
        'prune_crash_reports.h',
        'settings.cc',
        'settings.h',
        'simple_string_dictionary.h',
        'simple_address_range_bag.h',
        'simulate_crash.h',
        'simulate_crash_linux.h',
        'simulate_crash_mac.cc',
        'simulate_crash_mac.h',
        'simulate_crash_win.h',
      ],
      'conditions': [
        ['OS=="win"', {
          'link_settings': {
            'libraries': [
              '-lrpcrt4.lib',
            ],
          },
        }],
        ['OS=="linux" or OS=="android"', {
          'sources': [
            'client_argv_handling.cc',
            'client_argv_handling.h',
            'crashpad_info_note.S',
            'crash_report_database_generic.cc',
          ],
        }],
      ],
      'target_conditions': [
        ['OS=="android"', {
          'sources/': [
            ['include', '^crashpad_client_linux\\.cc$'],
            ['include', '^simulate_crash_linux\\.h$'],
          ],
        }],
      ],
      'direct_dependent_settings': {
        'include_dirs': [
          '..',
        ],
      },
    },
  ],
}
