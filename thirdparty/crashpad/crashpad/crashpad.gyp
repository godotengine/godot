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
  'targets': [
    {
      'target_name': 'All',
      'type': 'none',
      'dependencies': [
        'client/client.gyp:*',
        'client/client_test.gyp:*',
        'compat/compat.gyp:*',
        'handler/handler.gyp:*',
        'handler/handler_test.gyp:*',
        'minidump/minidump.gyp:*',
        'minidump/minidump_test.gyp:*',
        'snapshot/snapshot.gyp:*',
        'snapshot/snapshot_test.gyp:*',
        'test/test.gyp:*',
        'test/test_test.gyp:*',
        'tools/tools.gyp:*',
        'util/util.gyp:*',
        'util/util_test.gyp:*',
      ],
      'sources': [
        'doc/support/crashpad.doxy.h',
        'package.h',
      ],
      'conditions': [
        ['OS!="mac" and OS!="win"', {
          'suppress_wildcard': 1,
        }],
      ],
    },
  ],
}
