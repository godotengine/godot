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

vars = {
  'chromium_git': 'https://chromium.googlesource.com',
  'pull_linux_clang': False,
  'pull_win_toolchain': False
}

deps = {
  'buildtools':
      Var('chromium_git') + '/chromium/buildtools.git@' +
      '6fe4a3251488f7af86d64fc25cf442e817cf6133',
  'crashpad/third_party/gtest/gtest':
      Var('chromium_git') + '/external/github.com/google/googletest@' +
      'c091b0469ab4c04ee9411ef770f32360945f4c53',
  'crashpad/third_party/gyp/gyp':
      Var('chromium_git') + '/external/gyp@' +
      '5e2b3ddde7cda5eb6bc09a5546a76b00e49d888f',
  'crashpad/third_party/mini_chromium/mini_chromium':
      Var('chromium_git') + '/chromium/mini_chromium@' +
      '8d641e30a8b12088649606b912c2bc4947419ccc',
  'crashpad/third_party/libfuzzer/src':
      Var('chromium_git') + '/chromium/llvm-project/compiler-rt/lib/fuzzer.git@' +
      'fda403cf93ecb8792cb1d061564d89a6553ca020',
  'crashpad/third_party/zlib/zlib':
      Var('chromium_git') + '/chromium/src/third_party/zlib@' +
      '13dc246a58e4b72104d35f9b1809af95221ebda7',

  # CIPD packages below.
  'crashpad/third_party/linux/clang/linux-amd64': {
    'packages': [
      {
        'package': 'fuchsia/clang/linux-amd64',
        'version': 'goma',
      },
    ],
    'condition': 'checkout_linux and pull_linux_clang',
    'dep_type': 'cipd'
  },
  'crashpad/third_party/linux/clang/mac-amd64': {
    'packages': [
      {
        'package': 'fuchsia/clang/mac-amd64',
        'version': 'goma',
      },
    ],
    'condition': 'checkout_fuchsia and host_os == "mac"',
    'dep_type': 'cipd'
  },
  'crashpad/third_party/fuchsia/clang/linux-amd64': {
    'packages': [
      {
        'package': 'fuchsia/clang/linux-amd64',
        'version': 'goma',
      },
    ],
    'condition': 'checkout_fuchsia and host_os == "linux"',
    'dep_type': 'cipd'
  },
  'crashpad/third_party/fuchsia/qemu/mac-amd64': {
    'packages': [
      {
        'package': 'fuchsia/qemu/mac-amd64',
        'version': 'latest'
      },
    ],
    'condition': 'checkout_fuchsia and host_os == "mac"',
    'dep_type': 'cipd'
  },
  'crashpad/third_party/fuchsia/qemu/linux-amd64': {
    'packages': [
      {
        'package': 'fuchsia/qemu/linux-amd64',
        'version': 'latest'
      },
    ],
    'condition': 'checkout_fuchsia and host_os == "linux"',
    'dep_type': 'cipd'
  },
  'crashpad/third_party/fuchsia/sdk/linux-amd64': {
    # The SDK is keyed to the host system because it contains build tools.
    # Currently, linux-amd64 is the only SDK published (see
    # https://chrome-infra-packages.appspot.com/#/?path=fuchsia/sdk).
    # As long as this is the case, use that SDK package
    # even on other build hosts.
    # The sysroot (containing headers and libraries) and other components are
    # related to the target and should be functional with an appropriate
    # toolchain that runs on the build host (fuchsia_clang, above).
    'packages': [
      {
        'package': 'fuchsia/sdk/linux-amd64',
        'version': 'latest'
      },
    ],
    'condition': 'checkout_fuchsia',
    'dep_type': 'cipd'
  },
  'crashpad/third_party/win/toolchain': {
    # This package is only updated when the solution in .gclient includes an
    # entry like:
    #   "custom_vars": { "pull_win_toolchain": True }
    # This is because the contained bits are not redistributable.
    'packages': [
      {
        'package': 'chrome_internal/third_party/sdk/windows',
        'version': 'uploaded:2018-06-13'
      },
    ],
    'condition': 'checkout_win and pull_win_toolchain',
    'dep_type': 'cipd'
  },
}

hooks = [
  {
    'name': 'clang_format_mac',
    'pattern': '.',
    'condition': 'host_os == "mac"',
    'action': [
      'download_from_google_storage',
      '--no_resume',
      '--no_auth',
      '--bucket=chromium-clang-format',
      '--sha1_file',
      'buildtools/mac/clang-format.sha1',
    ],
  },
  {
    'name': 'clang_format_linux',
    'pattern': '.',
    'condition': 'host_os == "linux"',
    'action': [
      'download_from_google_storage',
      '--no_resume',
      '--no_auth',
      '--bucket=chromium-clang-format',
      '--sha1_file',
      'buildtools/linux64/clang-format.sha1',
    ],
  },
  {
    'name': 'clang_format_win',
    'pattern': '.',
    'condition': 'host_os == "win"',
    'action': [
      'download_from_google_storage',
      '--no_resume',
      '--no_auth',
      '--bucket=chromium-clang-format',
      '--sha1_file',
      'buildtools/win/clang-format.exe.sha1',
    ],
  },
  {
    'name': 'gn_mac',
    'pattern': '.',
    'condition': 'host_os == "mac"',
    'action': [
      'download_from_google_storage',
      '--no_resume',
      '--no_auth',
      '--bucket=chromium-gn',
      '--sha1_file',
      'buildtools/mac/gn.sha1',
    ],
  },
  {
    'name': 'gn_linux',
    'pattern': '.',
    'condition': 'host_os == "linux"',
    'action': [
      'download_from_google_storage',
      '--no_resume',
      '--no_auth',
      '--bucket=chromium-gn',
      '--sha1_file',
      'buildtools/linux64/gn.sha1',
    ],
  },
  {
    'name': 'gn_win',
    'pattern': '.',
    'condition': 'host_os == "win"',
    'action': [
      'download_from_google_storage',
      '--no_resume',
      '--no_auth',
      '--bucket=chromium-gn',
      '--sha1_file',
      'buildtools/win/gn.exe.sha1',
    ],
  },
  {
    # If using a local clang ("pull_linux_clang" above), also pull down a
    # sysroot.
    'name': 'sysroot_linux',
    'pattern': '.',
    'condition': 'checkout_linux and pull_linux_clang',
    'action': [
      'crashpad/build/install_linux_sysroot.py',
    ],
  },
  {
    'name': 'gyp',
    'pattern': '\.gypi?$',
    'action': ['python', 'crashpad/build/gyp_crashpad.py'],
  },
]

recursedeps = [
  'buildtools',
]
