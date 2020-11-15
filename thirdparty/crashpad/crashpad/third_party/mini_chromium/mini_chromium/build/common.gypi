# Copyright 2009 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

{
  'variables': {
    'variables': {
      'clang%': 0,
      'conditions': [
        ['OS=="mac"', {
          'clang%': 1,
        }],
      ],
    },
    'clang%': '<(clang)',

    'android_api_level%': '',

    'mac_sdk%': '',
    'mac_deployment_target%': '',

    'target_arch%': 'x64',
  },

  'target_defaults': {
    'includes': [
      'filename_rules.gypi',
    ],
    'conditions': [

      ['OS=="mac"', {
        'xcode_settings': {
          'ALWAYS_SEARCH_USER_PATHS': 'NO',
          'GCC_C_LANGUAGE_STANDARD': 'c99',  # -std=c99
          'GCC_CW_ASM_SYNTAX': 'NO',  # No -fasm-blocks
          'GCC_DYNAMIC_NO_PIC': 'NO',  # No -mdynamic-no-pic
          'GCC_ENABLE_CPP_EXCEPTIONS': 'NO',  # -fno-exceptions
          'GCC_ENABLE_CPP_RTTI': 'NO',  # -fno-rtti
          'GCC_ENABLE_PASCAL_STRINGS': 'NO',  # No -mpascal-strings

          # GCC_INLINES_ARE_PRIVATE_EXTERN maps to -fvisibility-inlines-hidden
          'GCC_INLINES_ARE_PRIVATE_EXTERN': 'YES',

          'GCC_OBJC_CALL_CXX_CDTORS': 'YES',  # -fobjc-call-cxx-cdtors
          'GCC_PRECOMPILE_PREFIX_HEADER': 'NO',
          'GCC_SYMBOLS_PRIVATE_EXTERN': 'YES',  # -fvisibility=hidden
          'GCC_TREAT_WARNINGS_AS_ERRORS': 'YES',  # -Werror
          'GCC_WARN_ABOUT_MISSING_NEWLINE': 'YES',  # -Wnewline-eof
          'OTHER_CFLAGS': [
            '-fno-strict-aliasing',  # See http://crbug.com/32204
            '-fstack-protector-all',  # Implies -fstack-protector
          ],
          'USE_HEADERMAP': 'NO',
          'WARNING_CFLAGS': [
            '-Wall',
            '-Wendif-labels',
            '-Wextra',

            # Don't warn about unused function parameters.
            '-Wno-unused-parameter',

            # Don't warn about the "struct foo f = {0};" initialization
            # pattern.
            '-Wno-missing-field-initializers',
          ],

          'conditions': [
            ['clang!=0', {
              'CLANG_CXX_LANGUAGE_STANDARD': 'c++14',  # -std=c++14

              # Don't link in libarclite_macosx.a, see http://crbug.com/156530.
              'CLANG_LINK_OBJC_RUNTIME': 'NO',  # No -fobjc-link-runtime

              # CLANG_WARN_OBJC_MISSING_PROPERTY_SYNTHESIS maps to
              # -Wobjc-missing-property-synthesis
              'CLANG_WARN_OBJC_MISSING_PROPERTY_SYNTHESIS': 'YES',

              'GCC_VERSION': 'com.apple.compilers.llvm.clang.1_0',
              'WARNING_CFLAGS': [
                '-Wexit-time-destructors',
                '-Wheader-hygiene',
                '-Wimplicit-fallthrough',
                '-Wno-selector-type-mismatch',
                '-Wsign-compare',
                '-Wstring-conversion',
              ],
            }, {  # else: clang==0
              'GCC_VERSION': '4.2',
            }],

            ['mac_sdk!=""', {
              'SDKROOT': 'macosx<(mac_sdk)',  # -isysroot
            }],

            ['mac_deployment_target!=""', {
              # MACOSX_DEPLOYMENT_TARGET maps to -mmacosx-version-min
              'MACOSX_DEPLOYMENT_TARGET': '<(mac_deployment_target)',
            }],

            ['target_arch=="ia32"', {
              'ARCHS': [
                'i386',
              ],
            }],
            ['target_arch=="x64"', {
              'ARCHS': [
                'x86_64',
              ],
            }],
          ],

          'target_conditions': [
            ['_type=="executable"', {
              'OTHER_LDFLAGS': [
                '-Wl,-pie',  # Position-independent executable (MH_PIE)
              ],
            }],
          ],

        },
      }],

      ['OS=="linux" or OS=="android"', {
        'cflags': [
          '-fPIC',
          '-fno-exceptions',
          '-fno-strict-aliasing',  # See http://crbug.com/32204
          '-fstack-protector-all',  # Implies -fstack-protector
          '-fvisibility=hidden',
          '-g',
          '-pipe',
          '-pthread',
          '-Wall',
          '-Werror',
          '-Wextra',
          '-Wno-unused-parameter',
          '-Wno-missing-field-initializers',
        ],
        'cflags_cc': [
          '-fno-rtti',
          '-fvisibility-inlines-hidden',
          '-std=c++14',
        ],
        'defines': [
          '_FILE_OFFSET_BITS=64',
        ],
        'ldflags': [
          '-fPIC',
          '-pthread',
          '-Wl,--as-needed',
          '-Wl,-z,noexecstack',
        ],

        'conditions': [
          ['clang!=0', {
            'cflags': [
              '-Wexit-time-destructors',
              '-Wheader-hygiene',
              '-Wimplicit-fallthrough',
              '-Wsign-compare',
              '-Wstring-conversion',
            ],
          }, {  # else: clang==0
            'conditions': [
              ['target_arch=="ia32"', {
                'cflags': [
                  '-msse2',
                  '-mfpmath=sse',
                ],
              }],
            ],
          }],

          ['OS=="linux"', {
            'conditions': [
              ['target_arch=="ia32"', {
                'cflags': [
                  '-m32',
                ],
                'ldflags': [
                  '-m32',
                ],
              }],
              ['target_arch=="x64"', {
                'cflags': [
                  '-m64',
                ],
                'ldflags': [
                  '-m64',
                ],
              }],
            ],
          }],

          ['OS=="android"', {
            'conditions': [
              ['android_api_level!=""', {
                'defines': [
                  # With deprecated headers, this was available by #including
                  # <android/api-level.h>, but with unified headers, the desired
                  # value must be pushed into the build from the outside when
                  # building with GCC. See
                  # https://android.googlesource.com/platform/ndk/+/master/docs/UnifiedHeaders.md.
                  # It’s harmless to define this when building with Clang.
                  '__ANDROID_API__=<(android_api_level)',
                ],
              }],
            ],
          }],
        ],

        'target_conditions': [
          ['_type=="executable"', {
            'ldflags': [
              '-pie',
            ],
          }],
        ],
      }],

      ['OS=="win"', {
        'msvs_configuration_attributes': {
          'CharacterSet': '1',
        },
        'msvs_settings': {
          'VCCLCompilerTool': {
            'WarningLevel': '4',
            'WarnAsError': 'true',
            'DebugInformationFormat': '3',
            'ExceptionHandling': '0',
            'RuntimeTypeInfo': 'false',
            'BufferSecurityCheck': 'true',
            'EnableFunctionLevelLinking': 'true',
            'AdditionalOptions': [
              '/bigobj',  # Maximum 2^32 sections in .obj files (default 2^16).
            ],
          },
          'VCLinkerTool': {
            'GenerateDebugInformation': 'true',
            'RandomizedBaseAddress': '2',  # /DYNAMICBASE.
            'SubSystem': '1',
          },
        },
        'msvs_disabled_warnings': [
          4100,  # Unreferenced formal parameter.
          4127,  # Conditional expression is constant.
          4351,  # New behavior: elements of array will be default initialized.
          4530,  # Exceptions are disabled.
          4702,  # Unreachable code. https://crbug.com/346399
          4996,  # 'X' was declared deprecated.
        ],
        'defines': [
          '_HAS_EXCEPTIONS=0',
          '_CRT_SECURE_NO_WARNINGS',
          'NOMINMAX',
          'WIN32_LEAN_AND_MEAN',
        ],
      }],

    ],
    'default_configuration': 'Debug',
    'configurations': {
      'Release': {
        'defines': [
          'NDEBUG',
        ],
        'conditions': [

          ['OS=="mac"', {
            'xcode_settings': {
              'DEAD_CODE_STRIPPING': 'YES',  # -Wl,-dead_strip
              'DEBUG_INFORMATION_FORMAT': 'dwarf-with-dsym',
              'GCC_OPTIMIZATION_LEVEL': '3',

              'target_conditions': [
                ['_type=="executable" or _type=="shared_library" or \
                  _type=="loadable_module"', {
                  'DEPLOYMENT_POSTPROCESSING': 'YES',
                  'STRIP_INSTALLED_PRODUCT': 'YES',
                }],
                ['_type=="shared_library" or _type=="loadable_module"', {
                  'STRIPFLAGS': '-x',
                }],
              ],

            },
          }],

          ['OS=="linux" or OS=="android"', {
            'cflags': [
              '-O3',
              '-fdata-sections',
              '-ffunction-sections',
            ],
            'ldflags': [
              '-Wl,-O1',
              '-Wl,--gc-sections',
            ],

            'conditions': [
              ['clang==0', {
                'cflags': [
                  '-fno-ident',
                ],
              }],
            ],

          }],

          ['OS=="win"', {
            'msvs_configuration_platform': 'Win32',
            'msvs_settings': {
              'VCCLCompilerTool': {
                'RuntimeLibrary': '0',  # /MT.
                'Optimization': '3',
                'AdditionalOptions': [
                  '/Zo',  # Improve debugging optimized builds.
                ],
              },
              'VCLibrarianTool': {
                'TargetMachine': '1',  # x86.
              },
              'VCLinkerTool': {
                'MinimumRequiredVersion': '5.01',  # XP.
                'TargetMachine': '1',  # x86.
              },
            },
          }],

        ],
      },
      'Debug': {
        'conditions': [

          ['OS=="mac"', {
            'xcode_settings': {
              'COPY_PHASE_STRIP': 'NO',
              'DEBUG_INFORMATION_FORMAT': 'dwarf',
              'GCC_OPTIMIZATION_LEVEL': '0',
            },
          }],

          ['OS=="linux" or OS=="android"', {
            'cflags': [
              '-O0',
            ],
          }],

          ['OS=="win"', {
            'msvs_configuration_platform': 'Win32',
            'msvs_settings': {
              'VCCLCompilerTool': {
                'RuntimeLibrary': '1',  # /MTd.
                'Optimization': '0',
              },
              'VCLibrarianTool': {
                'TargetMachine': '1',  # x86.
              },
              'VCLinkerTool': {
                'MinimumRequiredVersion': '5.01',  # XP.
                'TargetMachine': '1',  # x86.
              },
            },
            'defines': [
              '_DEBUG',
              '_ITERATOR_DEBUG_LEVEL=2',
            ],
          }],

        ],
      },

      'conditions': [
        ['OS=="win"', {
          # gyp-ninja seems to require these, but we don't use them.
          'Debug_x64': {
            'inherit_from': ['Debug'],
            'msvs_configuration_platform': 'x64',
            'msvs_settings': {
              'VCLibrarianTool': {
                'TargetMachine': '17',  # x64.
              },
              'VCLinkerTool': {
                'MinimumRequiredVersion': '5.02',  # Server 2003.
                'TargetMachine': '17',  # x64.
              },
            },
          },
          'Release_x64': {
            'inherit_from': ['Release'],
            'msvs_configuration_platform': 'x64',
            'msvs_settings': {
              'VCLibrarianTool': {
                'TargetMachine': '17',  # x64.
              },
              'VCLinkerTool': {
                'MinimumRequiredVersion': '5.02',  # Server 2003.
                'TargetMachine': '17',  # x64.
              },
            },
          }
        }],
      ],
    },
  },

  'conditions': [
    ['OS=="mac"', {
      'xcode_settings': {
        'SYMROOT': '<(DEPTH)/xcodebuild',
      },
    }],
  ],
}
