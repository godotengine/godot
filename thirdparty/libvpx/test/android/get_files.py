# Copyright (c) 2013 The WebM project authors. All Rights Reserved.
#
# Use of this source code is governed by a BSD-style license
# that can be found in the LICENSE file in the root of the source
# tree. An additional intellectual property rights grant can be found
# in the file PATENTS.  All contributing project authors may
# be found in the AUTHORS file in the root of the source tree.
#
# This simple script pulls test files from the webm homepage
# It is intelligent enough to only pull files if
#   1) File / test_data folder does not exist
#   2) SHA mismatch

import pycurl
import csv
import hashlib
import re
import os.path
import time
import itertools
import sys
import getopt

#globals
url = ''
file_list_path = ''
local_resource_path = ''

# Helper functions:
# A simple function which returns the sha hash of a file in hex
def get_file_sha(filename):
  try:
    sha_hash = hashlib.sha1()
    with open(filename, 'rb') as file:
      buf = file.read(HASH_CHUNK)
      while len(buf) > 0:
        sha_hash.update(buf)
        buf = file.read(HASH_CHUNK)
      return sha_hash.hexdigest()
  except IOError:
    print("Error reading " + filename)

# Downloads a file from a url, and then checks the sha against the passed
# in sha
def download_and_check_sha(url, filename, sha):
  path = os.path.join(local_resource_path, filename)
  fp = open(path, "wb")
  curl = pycurl.Curl()
  curl.setopt(pycurl.URL, url + "/" + filename)
  curl.setopt(pycurl.WRITEDATA, fp)
  curl.perform()
  curl.close()
  fp.close()
  return get_file_sha(path) == sha

#constants
ftp_retries = 3

SHA_COL = 0
NAME_COL = 1
EXPECTED_COL = 2
HASH_CHUNK = 65536

# Main script
try:
  opts, args = \
      getopt.getopt(sys.argv[1:], \
                    "u:i:o:", ["url=", "input_csv=", "output_dir="])
except:
  print('get_files.py -u <url> -i <input_csv> -o <output_dir>')
  sys.exit(2)

for opt, arg in opts:
  if opt == '-u':
    url = arg
  elif opt in ("-i", "--input_csv"):
    file_list_path = os.path.join(arg)
  elif opt in ("-o", "--output_dir"):
    local_resource_path = os.path.join(arg)

if len(sys.argv) != 7:
  print("Expects two paths and a url!")
  exit(1)

if not os.path.isdir(local_resource_path):
  os.makedirs(local_resource_path)

file_list_csv = open(file_list_path, "rb")

# Our 'csv' file uses multiple spaces as a delimiter, python's
# csv class only uses single character delimiters, so we convert them below
file_list_reader = csv.reader((re.sub(' +', ' ', line.decode('utf-8')) \
    for line in file_list_csv), delimiter = ' ')

file_shas = []
file_names = []

for row in file_list_reader:
  if len(row) != EXPECTED_COL:
      continue
  file_shas.append(row[SHA_COL])
  file_names.append(row[NAME_COL])

file_list_csv.close()

# Download files, only if they don't already exist and have correct shas
for filename, sha in zip(file_names, file_shas):
  filename = filename.lstrip('*')
  path = os.path.join(local_resource_path, filename)
  if os.path.isfile(path) \
      and get_file_sha(path) == sha:
    print(path + ' exists, skipping')
    continue
  for retry in range(0, ftp_retries):
    print("Downloading " + path)
    if not download_and_check_sha(url, filename, sha):
      print("Sha does not match, retrying...")
    else:
      break
