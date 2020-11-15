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
    '../../build/crashpad_dependencies.gypi',
  ],
  'conditions': [
    ['1==1', {  # Defer processing until crashpad_dependencies is set
      'variables': {
        'conditions': [
          ['crashpad_dependencies=="external"', {
            'zlib_source%': 'external',
          }, 'OS!="win"', {
            # Use the system zlib by default where available, as it is on most
            # platforms. Windows does not have a system zlib, so use “embedded”
            # which directs the build to use the source code in the zlib
            # subdirectory.
            'zlib_source%': 'system',
          }, {
            'zlib_source%': 'embedded',
          }],
        ],
      },
    }],
  ],
  'targets': [
    {
      'target_name': 'zlib',
      'conditions': [
        ['zlib_source=="system"', {
          'type': 'none',
          'direct_dependent_settings': {
            'defines': [
              'CRASHPAD_ZLIB_SOURCE_SYSTEM',
            ],
          },
          'link_settings': {
            'conditions': [
              ['OS=="mac"', {
                'libraries': [
                  '$(SDKROOT)/usr/lib/libz.dylib',
                ],
              }, {
                'libraries': [
                  '-lz',
                ],
              }],
            ],
          },
        }],
        ['zlib_source=="embedded"', {
          'type': 'static_library',
          'include_dirs': [
              'zlib',
          ],
          'defines': [
            'CRASHPAD_ZLIB_SOURCE_EMBEDDED',
            'HAVE_STDARG_H',
          ],
          'direct_dependent_settings': {
            'include_dirs': [
              'zlib',
            ],
            'defines': [
              'CRASHPAD_ZLIB_SOURCE_EMBEDDED',
            ],
          },
          'sources': [
            'zlib/adler32.c',
            'zlib/compress.c',
            'zlib/crc32.c',
            'zlib/crc32.h',
            'zlib/crc_folding.c',
            'zlib/deflate.c',
            'zlib/deflate.h',
            'zlib/fill_window_sse.c',
            'zlib/gzclose.c',
            'zlib/gzguts.h',
            'zlib/gzlib.c',
            'zlib/gzread.c',
            'zlib/gzwrite.c',
            'zlib/infback.c',
            'zlib/inffast.c',
            'zlib/inffast.h',
            'zlib/inffixed.h',
            'zlib/inflate.c',
            'zlib/inflate.h',
            'zlib/inftrees.c',
            'zlib/inftrees.h',
            'zlib/names.h',
            'zlib/simd_stub.c',
            'zlib/trees.c',
            'zlib/trees.h',
            'zlib/uncompr.c',
            'zlib/x86.c',
            'zlib/x86.h',
            'zlib/zconf.h',
            'zlib/zlib.h',
            'zlib/zutil.c',
            'zlib/zutil.h',
            'zlib_crashpad.h',
          ],
          'conditions': [
            ['target_arch=="ia32" or target_arch=="x64"', {
              'sources!': [
                'zlib/simd_stub.c',
              ],
              'cflags': [
                '-msse4.2',
                '-mpclmul',
              ],
              'xcode_settings': {
                'OTHER_CFLAGS': [
                  '-msse4.2',
                  '-mpclmul',
                ],
              },
            }, {
              'sources!': [
                'zlib/crc_folding.c',
                'zlib/fill_window_sse.c',
                'zlib/x86.c',
                'zlib/x86.h',
              ],
            }],
            ['OS!="win"', {
              'defines': [
                'HAVE_HIDDEN',
                'HAVE_UNISTD_H',
              ],
            }, {
              'msvs_disabled_warnings': [
                4131,  # uses old-style declarator
                4244,  # conversion from 't1' to 't2', possible loss of data
                4245,  # conversion from 't1' to 't2', signed/unsigned mismatch
                4267,  # conversion from 'size_t' to 't', possible loss of data
                4324,  # structure was padded due to alignment specifier
              ],
            }],
          ],
        }],
        ['zlib_source=="external"', {
          'type': 'none',
          'direct_dependent_settings': {
            'defines': [
              'CRASHPAD_ZLIB_SOURCE_EXTERNAL',
            ],
          },
          'dependencies': [
            '../../../../zlib/zlib.gyp:zlib',
          ],
        }],
      ],
    },
  ],
}
