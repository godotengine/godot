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
      'target_name': 'crashpad_minidump',
      'type': 'static_library',
      'dependencies': [
        '../compat/compat.gyp:crashpad_compat',
        '../snapshot/snapshot.gyp:crashpad_snapshot',
        '../third_party/mini_chromium/mini_chromium.gyp:base',
        '../util/util.gyp:crashpad_util',
      ],
      'export_dependent_settings': [
        '../compat/compat.gyp:crashpad_compat',
      ],
      'include_dirs': [
        '..',
      ],
      'sources': [
        'minidump_annotation_writer.cc',
        'minidump_annotation_writer.h',
        'minidump_byte_array_writer.cc',
        'minidump_byte_array_writer.h',
        'minidump_context.h',
        'minidump_context_writer.cc',
        'minidump_context_writer.h',
        'minidump_crashpad_info_writer.cc',
        'minidump_crashpad_info_writer.h',
        'minidump_exception_writer.cc',
        'minidump_exception_writer.h',
        'minidump_extensions.cc',
        'minidump_extensions.h',
        'minidump_file_writer.cc',
        'minidump_file_writer.h',
        'minidump_handle_writer.cc',
        'minidump_handle_writer.h',
        'minidump_memory_info_writer.cc',
        'minidump_memory_info_writer.h',
        'minidump_memory_writer.cc',
        'minidump_memory_writer.h',
        'minidump_misc_info_writer.cc',
        'minidump_misc_info_writer.h',
        'minidump_module_crashpad_info_writer.cc',
        'minidump_module_crashpad_info_writer.h',
        'minidump_module_writer.cc',
        'minidump_module_writer.h',
        'minidump_rva_list_writer.cc',
        'minidump_rva_list_writer.h',
        'minidump_simple_string_dictionary_writer.cc',
        'minidump_simple_string_dictionary_writer.h',
        'minidump_stream_writer.cc',
        'minidump_stream_writer.h',
        'minidump_string_writer.cc',
        'minidump_string_writer.h',
        'minidump_system_info_writer.cc',
        'minidump_system_info_writer.h',
        'minidump_thread_id_map.cc',
        'minidump_thread_id_map.h',
        'minidump_thread_writer.cc',
        'minidump_thread_writer.h',
        'minidump_unloaded_module_writer.cc',
        'minidump_unloaded_module_writer.h',
        'minidump_user_extension_stream_data_source.cc',
        'minidump_user_extension_stream_data_source.h',
        'minidump_user_stream_writer.cc',
        'minidump_user_stream_writer.h',
        'minidump_writable.cc',
        'minidump_writable.h',
        'minidump_writer_util.cc',
        'minidump_writer_util.h',
      ],
    },
  ],
}
