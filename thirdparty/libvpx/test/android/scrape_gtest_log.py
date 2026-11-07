# Copyright (c) 2014 The WebM project authors. All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the LICENSE file in the root of the source
# tree. An additional intellectual property rights grant can be found
# in the file PATENTS.  All contributing project authors may
# be found in the AUTHORS file in the root of the source tree.

"""Standalone script which parses a gtest log for json.

Json is returned returns as an array.  This script is used by the libvpx
waterfall to gather json results mixed in with gtest logs.  This is
dubious software engineering.
"""

import getopt
import json
import os
import re
import sys


def main():
  if len(sys.argv) != 3:
    print "Expects a file to write json to!"
    exit(1)

  try:
    opts, _ = \
        getopt.getopt(sys.argv[1:], \
                      'o:', ['output-json='])
  except getopt.GetOptError:
    print 'scrape_gtest_log.py -o <output_json>'
    sys.exit(2)

  output_json = ''
  for opt, arg in opts:
    if opt in ('-o', '--output-json'):
      output_json = os.path.join(arg)

  blob = sys.stdin.read()
  json_string = '[' + ','.join('{' + x + '}' for x in
                               re.findall(r'{([^}]*.?)}', blob)) + ']'
  print blob

  output = json.dumps(json.loads(json_string), indent=4, sort_keys=True)
  print output

  path = os.path.dirname(output_json)
  if path and not os.path.exists(path):
    os.makedirs(path)

  outfile = open(output_json, 'w')
  outfile.write(output)

if __name__ == '__main__':
  sys.exit(main())
