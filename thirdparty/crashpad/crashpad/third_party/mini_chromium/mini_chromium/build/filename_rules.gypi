# Copyright 2012 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

{
  'target_conditions': [
    ['OS!="mac"', {
      'sources/': [
        ['exclude', '_(cocoa|mac)(_test)?\\.(h|cc|mm?)$'],
        ['exclude', '(^|/)(cocoa|mac|mach)/'],
      ],
    }],
    ['OS!="linux"', {
      'sources/': [
        ['exclude', '_linux(_test)?\\.(h|cc)$'],
        ['exclude', '(^|/)linux/'],
      ],
    }],
    ['OS!="android"', {
      'sources/': [
        ['exclude', '_android(_test)?\\.(h|cc)$'],
        ['exclude', '(^|/)android/'],
      ],
    }],
    ['OS=="win"', {
      'sources/': [
        ['exclude', '_posix(_test)?\\.(h|cc)$'],
        ['exclude', '(^|/)posix/'],
      ],
    }, {
      'sources/': [
        ['exclude', '_win(_test)?\\.(h|cc)$'],
        ['exclude', '(^|/)win/'],
      ],
    }],
  ],
}
