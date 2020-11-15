# Copyright 2011 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

{
  'includes': [
    'build/common.gypi',
  ],

  'targets': [
    {
      'target_name': 'All',
      'type': 'none',
      'suppress_wildcard': 1,
      'dependencies': [
        'base/base.gyp:*',
      ],
    },
  ],
}
