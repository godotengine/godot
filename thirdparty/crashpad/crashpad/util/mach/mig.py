#!/usr/bin/env python
# coding: utf-8

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

import argparse
import os
import re
import subprocess
import sys

def FixUserImplementation(implementation):
  """Rewrites a MIG-generated user implementation (.c) file.

  Rewrites the file at |implementation| by adding “__attribute__((unused))” to
  the definition of any structure typedefed as “__Reply” by searching for the
  pattern unique to those structure definitions. These structures are in fact
  unused in the user implementation file, and this will trigger a
  -Wunused-local-typedefs warning in gcc unless removed or marked with the
  “unused” attribute.
  """

  file = open(implementation, 'r+')
  contents = file.read()

  pattern = re.compile('^(\t} __Reply);$', re.MULTILINE)
  contents = pattern.sub(r'\1 __attribute__((unused));', contents)

  file.seek(0)
  file.truncate()
  file.write(contents)
  file.close()

def FixServerImplementation(implementation):
  """Rewrites a MIG-generated server implementation (.c) file.

  Rewrites the file at |implementation| by replacing “mig_internal” with
  “mig_external” on functions that begin with “__MIG_check__”. This makes these
  functions available to other callers outside this file from a linkage
  perspective. It then returns, as a list of lines, declarations that can be
  added to a header file, so that other files that include that header file will
  have access to these declarations from a compilation perspective.
  """

  file = open(implementation, 'r+')
  contents = file.read()

  # Find interesting declarations.
  declaration_pattern = \
      re.compile('^mig_internal (kern_return_t __MIG_check__.*)$',
                 re.MULTILINE)
  declarations = declaration_pattern.findall(contents)

  # Remove “__attribute__((__unused__))” from the declarations, and call them
  # “mig_external” or “extern” depending on whether “mig_external” is defined.
  attribute_pattern = re.compile(r'__attribute__\(\(__unused__\)\) ')
  declarations = ['#ifdef mig_external\nmig_external\n#else\nextern\n#endif\n' +
                  attribute_pattern.sub('', x) +
                  ';\n' for x in declarations]

  # Rewrite the declarations in this file as “mig_external”.
  contents = declaration_pattern.sub(r'mig_external \1', contents);

  # Crashpad never implements the mach_msg_server() MIG callouts. To avoid
  # needing to provide stub implementations, set KERN_FAILURE as the RetCode
  # and abort().
  routine_callout_pattern = re.compile(
      r'OutP->RetCode = (([a-zA-Z0-9_]+)\(.+\));')
  routine_callouts = routine_callout_pattern.findall(contents)
  for routine in routine_callouts:
    contents = contents.replace(routine[0], 'KERN_FAILURE; abort()')

  # Include the header for abort().
  contents = '#include <stdlib.h>\n' + contents

  file.seek(0)
  file.truncate()
  file.write(contents)
  file.close()
  return declarations

def FixHeader(header, declarations=[]):
  """Rewrites a MIG-generated header (.h) file.

  Rewrites the file at |header| by placing it inside an “extern "C"” block, so
  that it declares things properly when included by a C++ compilation unit.
  |declarations| can be a list of additional declarations to place inside the
  “extern "C"” block after the original contents of |header|.
  """

  file = open(header, 'r+')
  contents = file.read()
  declarations_text = ''.join(declarations)
  contents = '''\
#ifdef __cplusplus
extern "C" {
#endif

%s
%s
#ifdef __cplusplus
}
#endif
''' % (contents, declarations_text)
  file.seek(0)
  file.truncate()
  file.write(contents)
  file.close()

def main(args):
  parser = argparse.ArgumentParser()
  parser.add_argument('--developer-dir', help='Path to Xcode')
  parser.add_argument('--sdk', help='Path to SDK')
  parser.add_argument('--include',
                      default=[],
                      action='append',
                      help='Additional include directory')
  parser.add_argument('defs')
  parser.add_argument('user_c')
  parser.add_argument('server_c')
  parser.add_argument('user_h')
  parser.add_argument('server_h')
  parsed = parser.parse_args(args)

  command = ['mig',
             '-user', parsed.user_c,
             '-server', parsed.server_c,
             '-header', parsed.user_h,
             '-sheader', parsed.server_h,
            ]
  if parsed.developer_dir is not None:
    os.environ['DEVELOPER_DIR'] = parsed.developer_dir
  if parsed.sdk is not None:
    command.extend(['-isysroot', parsed.sdk])
  for include in parsed.include:
    command.extend(['-I' + include])
  command.append(parsed.defs)
  subprocess.check_call(command)
  FixUserImplementation(parsed.user_c)
  server_declarations = FixServerImplementation(parsed.server_c)
  FixHeader(parsed.user_h)
  FixHeader(parsed.server_h, server_declarations)

if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
