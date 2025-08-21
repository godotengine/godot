#              Copyright Catch2 Authors
# Distributed under the Boost Software License, Version 1.0.
#   (See accompanying file LICENSE.txt or copy at
#        https://www.boost.org/LICENSE_1_0.txt)

# SPDX-License-Identifier: BSL-1.0

project(
  'catch2',
  'cpp',
  version: '3.9.1', # CML version placeholder, don't delete
  license: 'BSL-1.0',
  meson_version: '>=0.54.1',
)

subdir('src/catch2')
if get_option('tests')
  subdir('tests')
endif
