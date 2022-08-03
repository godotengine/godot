#! /usr/bin/env python

# Copyright (c) 2022 NVIDIA Corporation
# Reply-To: Allison Vacanti <alliepiper16@gmail.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Released under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.

help_text = """%(prog)s [reference.json compare.json | reference_dir/ compare_dir/]

This script:

1. Runs `top -bco RES`, continuously extracting the memory usage of each process.
2. If a process uses more than `log_threshold` GiB and exceeds any other recorded
   entry for the process, it is stored in `entries`.
3. When this script receives SIGINT, it writes two files:
  * `log_file` will contain all recorded max-memory-per-process entries
  * `fail_file` will contain all entries that exceed `fail_threshold`
"""

import argparse
import os
import re
import signal
import sys

from subprocess import Popen, PIPE, STDOUT

parser = argparse.ArgumentParser(prog='memmon.py', usage=help_text)
parser.add_argument('--log-threshold', type=float, dest='log_threshold',
                    default=0.5,
                    help='Logging threshold in GiB.')
parser.add_argument('--fail-threshold', type=float, dest='fail_threshold',
                    default=2,
                    help='Failure threshold in GiB.')
parser.add_argument('--log-file', type=str, dest='log_file', default='memmon_log',
                    help='Output file for log entries.')
args, unused = parser.parse_known_args()

entries = {}


def signal_handler(sig, frame):
    # Sort by mem:
    sortentries = sorted(entries.items(), key=lambda x: x[1], reverse=True)

    lf = open(args.log_file, "w")

    for com, mem in sortentries:
        status = "PASS"
        if mem >= args.fail_threshold:
            status = "FAIL"
        line = "%4s | %3.1f GiB | %s\n" % (status, mem, com)
        lf.write(line)

    lf.close()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

# Find the toprc config file and configure top's env.
# This config:
# - Hides all columns except for RES and COMMAND
# - Sorts by RES
# - Enables long command strings (-c)
script_dir = os.path.dirname(os.path.realpath(__file__))
config_dir = os.path.join(script_dir, 'memmon_config')

proc = Popen(["top", "-b", "-w", "512"],
             stdin=PIPE, stdout=PIPE, stderr=STDOUT,
             env={"XDG_CONFIG_HOME": config_dir})

regex = re.compile("^\\s*([0-9.]+[kmgtp]?)\\s+(.+)\\s*$")


# Convert a memory string from top into floating point GiB
def parse_mem(mem_str):
    if mem_str[-1] == "k":
        return float(mem_str[:-1]) / (1024 * 1024)
    elif mem_str[-1] == "m":
        return float(mem_str[:-1]) / (1024)
    elif mem_str[-1] == "g":
        return float(mem_str[:-1])
    elif mem_str[-1] == "t":
        return float(mem_str[:-1]) * 1024
    elif mem_str[-1] == "p":  # please no
        return float(mem_str[:-1]) * 1024 * 1024
    # bytes:
    return float(mem_str) / (1024 * 1024 * 1024)


for line in proc.stdout:
    line = line.decode()
    match = regex.match(line)
    if match:
        mem = parse_mem(match.group(1))
        if mem < args.log_threshold and mem < args.fail_threshold:
            continue
        com = match.group(2)
        if com in entries and entries[com] > mem:
            continue
        if mem >= args.fail_threshold:
            # Print a notice immediately -- this helps identify the failures
            # as they happen, since `com` may not provide enough info.
            print("memmon.py failure: Build step exceed memory threshold:\n"
                  "  - Threshold: %3.1f GiB\n"
                  "  - Usage:     %3.1f GiB\n"
                  "  - Command:   %s" % (args.fail_threshold, mem, com))
        entries[com] = mem
