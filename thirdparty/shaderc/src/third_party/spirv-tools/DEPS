use_relative_paths = True

vars = {
  'chromium_git': 'https://chromium.googlesource.com',
  'github': 'https://github.com',

  'build_revision': '037f38ae0fe5e11b4f7c33b750fd7a1e9634a606',
  'buildtools_revision': 'ab7b6a7b350dd15804c87c20ce78982811fdd76f',
  'clang_revision': 'abe5e4f9dc0f1df848c7a0efa05256253e77a7b7',
  'effcee_revision': '04b624799f5a9dbaf3fa1dbed2ba9dce2fc8dcf2',
  'googletest_revision': '98a0d007d7092b72eea0e501bb9ad17908a1a036',
  'testing_revision': '340252637e2e7c72c0901dcbeeacfff419e19b59',
  're2_revision': '6cf8ccd82dbaab2668e9b13596c68183c9ecd13f',
  'spirv_headers_revision': 'e74c389f81915d0a48d6df1af83c3862c5ad85ab',
}

deps = {
  "build":
    Var('chromium_git') + "/chromium/src/build.git@" + Var('build_revision'),

  'buildtools':
      Var('chromium_git') + '/chromium/buildtools.git@' +
          Var('buildtools_revision'),

  'external/spirv-headers':
      Var('github') +  '/KhronosGroup/SPIRV-Headers.git@' +
          Var('spirv_headers_revision'),

  'external/googletest':
      Var('github') + '/google/googletest.git@' + Var('googletest_revision'),

  'external/effcee':
      Var('github') + '/google/effcee.git@' + Var('effcee_revision'),

  'external/re2':
      Var('github') + '/google/re2.git@' + Var('re2_revision'),

  'testing':
      Var('chromium_git') + '/chromium/src/testing@' +
          Var('testing_revision'),

  'tools/clang':
      Var('chromium_git') + '/chromium/src/tools/clang@' + Var('clang_revision')
}

recursedeps = [
  # buildtools provides clang_format, libc++, and libc++api
  'buildtools',
]

hooks = [
  {
    'name': 'gn_win',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--platform=win32',
                '--no_auth',
                '--bucket', 'chromium-gn',
                '-s', 'SPIRV-Tools/buildtools/win/gn.exe.sha1',
    ],
  },
  {
    'name': 'gn_mac',
    'pattern': '.',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--platform=darwin',
                '--no_auth',
                '--bucket', 'chromium-gn',
                '-s', 'SPIRV-Tools/buildtools/mac/gn.sha1',
    ],
  },
  {
    'name': 'gn_linux64',
    'pattern': '.',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--platform=linux*',
                '--no_auth',
                '--bucket', 'chromium-gn',
                '-s', 'SPIRV-Tools/buildtools/linux64/gn.sha1',
    ],
  },
  # Pull clang-format binaries using checked-in hashes.
  {
    'name': 'clang_format_win',
    'pattern': '.',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--platform=win32',
                '--no_auth',
                '--bucket', 'chromium-clang-format',
                '-s', 'SPIRV-Tools/buildtools/win/clang-format.exe.sha1',
    ],
  },
  {
    'name': 'clang_format_mac',
    'pattern': '.',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--platform=darwin',
                '--no_auth',
                '--bucket', 'chromium-clang-format',
                '-s', 'SPIRV-Tools/buildtools/mac/clang-format.sha1',
    ],
  },
  {
    'name': 'clang_format_linux',
    'pattern': '.',
    'action': [ 'download_from_google_storage',
                '--no_resume',
                '--platform=linux*',
                '--no_auth',
                '--bucket', 'chromium-clang-format',
                '-s', 'SPIRV-Tools/buildtools/linux64/clang-format.sha1',
    ],
  },
  {
    # Pull clang
    'name': 'clang',
    'pattern': '.',
    'action': ['python',
               'SPIRV-Tools/tools/clang/scripts/update.py'
    ],
  },
  {
    'name': 'sysroot_arm',
    'pattern': '.',
    'condition': 'checkout_linux and checkout_arm',
    'action': ['python', 'SPIRV-Tools/build/linux/sysroot_scripts/install-sysroot.py',
               '--arch=arm'],
  },
  {
    'name': 'sysroot_arm64',
    'pattern': '.',
    'condition': 'checkout_linux and checkout_arm64',
    'action': ['python', 'SPIRV-Tools/build/linux/sysroot_scripts/install-sysroot.py',
               '--arch=arm64'],
  },
  {
    'name': 'sysroot_x86',
    'pattern': '.',
    'condition': 'checkout_linux and (checkout_x86 or checkout_x64)',
    'action': ['python', 'SPIRV-Tools/build/linux/sysroot_scripts/install-sysroot.py',
               '--arch=x86'],
  },
  {
    'name': 'sysroot_mips',
    'pattern': '.',
    'condition': 'checkout_linux and checkout_mips',
    'action': ['python', 'SPIRV-Tools/build/linux/sysroot_scripts/install-sysroot.py',
               '--arch=mips'],
  },
  {
    'name': 'sysroot_x64',
    'pattern': '.',
    'condition': 'checkout_linux and checkout_x64',
    'action': ['python', 'SPIRV-Tools/build/linux/sysroot_scripts/install-sysroot.py',
               '--arch=x64'],
  },
  {
    # Update the Windows toolchain if necessary.
    'name': 'win_toolchain',
    'pattern': '.',
    'condition': 'checkout_win',
    'action': ['python', 'SPIRV-Tools/build/vs_toolchain.py', 'update', '--force'],
  },
  {
    # Update the Mac toolchain if necessary.
    'name': 'mac_toolchain',
    'pattern': '.',
    'action': ['python', 'SPIRV-Tools/build/mac_toolchain.py'],
  },
]
