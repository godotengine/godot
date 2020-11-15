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
      'target_name': 'crashpad_util_test',
      'type': 'executable',
      'dependencies': [
        'util.gyp:crashpad_util',
        '../client/client.gyp:crashpad_client',
        '../compat/compat.gyp:crashpad_compat',
        '../test/test.gyp:crashpad_gmock_main',
        '../test/test.gyp:crashpad_test',
        '../third_party/gtest/gmock.gyp:gmock',
        '../third_party/gtest/gtest.gyp:gtest',
        '../third_party/mini_chromium/mini_chromium.gyp:base',
        '../third_party/zlib/zlib.gyp:zlib',
      ],
      'include_dirs': [
        '..',
      ],
      'sources': [
        'file/delimited_file_reader_test.cc',
        'file/directory_reader_test.cc',
        'file/file_io_test.cc',
        'file/file_reader_test.cc',
        'file/filesystem_test.cc',
        'file/string_file_test.cc',
        'linux/auxiliary_vector_test.cc',
        'linux/memory_map_test.cc',
        'linux/proc_stat_reader_test.cc',
        'linux/ptrace_broker_test.cc',
        'linux/ptracer_test.cc',
        'linux/scoped_ptrace_attach_test.cc',
        'mac/launchd_test.mm',
        'mac/mac_util_test.mm',
        'mac/service_management_test.mm',
        'mac/xattr_test.cc',
        'mach/child_port_handshake_test.cc',
        'mach/child_port_server_test.cc',
        'mach/composite_mach_message_server_test.cc',
        'mach/exc_client_variants_test.cc',
        'mach/exc_server_variants_test.cc',
        'mach/exception_behaviors_test.cc',
        'mach/exception_ports_test.cc',
        'mach/exception_types_test.cc',
        'mach/mach_extensions_test.cc',
        'mach/mach_message_server_test.cc',
        'mach/mach_message_test.cc',
        'mach/notify_server_test.cc',
        'mach/scoped_task_suspend_test.cc',
        'mach/symbolic_constants_mach_test.cc',
        'mach/task_memory_test.cc',
        'misc/arraysize_unsafe_test.cc',
        'misc/capture_context_test.cc',
        'misc/capture_context_test_util.h',
        'misc/capture_context_test_util_linux.cc',
        'misc/capture_context_test_util_mac.cc',
        'misc/capture_context_test_util_win.cc',
        'misc/clock_test.cc',
        'misc/from_pointer_cast_test.cc',
        'misc/initialization_state_dcheck_test.cc',
        'misc/initialization_state_test.cc',
        'misc/paths_test.cc',
        'misc/scoped_forbid_return_test.cc',
        'misc/random_string_test.cc',
        'misc/range_set_test.cc',
        'misc/reinterpret_bytes_test.cc',
        'misc/time_test.cc',
        'misc/uuid_test.cc',
        'net/http_body_gzip_test.cc',
        'net/http_body_test.cc',
        'net/http_body_test_util.cc',
        'net/http_body_test_util.h',
        'net/http_multipart_builder_test.cc',
        'net/http_transport_test.cc',
        'net/url_test.cc',
        'numeric/checked_address_range_test.cc',
        'numeric/checked_range_test.cc',
        'numeric/in_range_cast_test.cc',
        'numeric/int128_test.cc',
        'posix/process_info_test.cc',
        'posix/scoped_mmap_test.cc',
        'posix/signals_test.cc',
        'posix/symbolic_constants_posix_test.cc',
        'process/process_memory_range_test.cc',
        'process/process_memory_test.cc',
        'stdlib/aligned_allocator_test.cc',
        'stdlib/map_insert_test.cc',
        'stdlib/string_number_conversion_test.cc',
        'stdlib/strlcpy_test.cc',
        'stdlib/strnlen_test.cc',
        'stdlib/thread_safe_vector_test.cc',
        'string/split_string_test.cc',
        'synchronization/semaphore_test.cc',
        'thread/thread_log_messages_test.cc',
        'thread/thread_test.cc',
        'thread/worker_thread_test.cc',
        'win/command_line_test.cc',
        'win/critical_section_with_debug_info_test.cc',
        'win/exception_handler_server_test.cc',
        'win/get_function_test.cc',
        'win/handle_test.cc',
        'win/initial_client_data_test.cc',
        'win/process_info_test.cc',
        'win/registration_protocol_win_test.cc',
        'win/safe_terminate_process_test.cc',
        'win/scoped_process_suspend_test.cc',
        'win/session_end_watcher_test.cc',
      ],
      'conditions': [
        ['OS=="mac"', {
          'link_settings': {
            'libraries': [
              '$(SDKROOT)/System/Library/Frameworks/Foundation.framework',
            ],
          },
        }],
        ['OS=="win"', {
          'dependencies': [
            'crashpad_util_test_process_info_test_child',
            'crashpad_util_test_safe_terminate_process_test_child',
          ],
          'link_settings': {
            'libraries': [
              '-ladvapi32.lib',
              '-limagehlp.lib',
              '-lrpcrt4.lib',
              '-luser32.lib',
            ],
          },
        }],
        ['OS=="android"', {
          # Things not yet ported to Android
          'sources/' : [
            ['exclude', '^net/http_transport_test\\.cc$'],
          ]
        }],
        ['OS!="linux" and OS!="android"', {
          'sources/': [
            ['exclude', '^process/'],
          ],
        }],
      ],
      'target_conditions': [
        ['OS=="android"', {
          'sources/': [
            ['include', '^linux/'],
            ['include', '^misc/capture_context_test_util_linux\\.cc$'],
          ],
        }],
      ],
    },
  ],
  'conditions': [
    ['OS!="android"', {
      'targets': [
        {
          'target_name': 'http_transport_test_server',
          'type': 'executable',
          'dependencies': [
            '../third_party/mini_chromium/mini_chromium.gyp:base',
            '../third_party/zlib/zlib.gyp:zlib',
            '../tools/tools.gyp:crashpad_tool_support',
            '../util/util.gyp:crashpad_util',
          ],
          'sources': [
            'net/http_transport_test_server.cc',
          ],
          'include_dirs': [
            '..',
          ],
          'xcode_settings': {
            'WARNING_CFLAGS!': [
              '-Wexit-time-destructors',
            ],
          },
          'cflags!': [
            '-Wexit-time-destructors',
          ],
          'conditions': [
            ['OS=="win"', {
              'link_settings': {
                'libraries': [
                  '-lws2_32.lib',
                ],
              },
            }],
          ],
        },
      ],
    }],
    ['OS=="win"', {
      'targets': [
        {
          'target_name': 'crashpad_util_test_process_info_test_child',
          'type': 'executable',
          'sources': [
            'win/process_info_test_child.cc',
          ],
        },
        {
          'target_name': 'crashpad_util_test_safe_terminate_process_test_child',
          'type': 'executable',
          'sources': [
            'win/safe_terminate_process_test_child.cc',
          ],
        },
      ]
    }],
  ],
}
