#! /usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# Copyright (c) 2018 NVIDIA Corporation
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
###############################################################################

from sys import exit 

from os.path import join, dirname, basename, realpath

from csv import DictReader as csv_dict_reader

from subprocess import Popen

from argparse import ArgumentParser as argument_parser

###############################################################################

def printable_cmd(c):
  """Converts a `list` of `str`s representing a shell command to a printable 
  `str`."""
  return " ".join(map(lambda e: '"' + str(e) + '"', c))

###############################################################################

def print_file(p):
  """Open the path `p` and print its contents to `stdout`."""
  print "********************************************************************************"
  with open(p) as f:
    for line in f:
      print line,
  print "********************************************************************************"

###############################################################################

ap = argument_parser(
  description = (
    "CUDA Eris driver script: runs a benchmark suite multiple times, combines "
    "the results, and outputs them in the CUDA Eris performance result format."
  )
)

ap.add_argument(
  "-b", "--benchmark", 
  help = ("The location of the benchmark suite executable to run."),
  type = str,
  default = join(dirname(realpath(__file__)), "bench"), 
  metavar = "R"
)

ap.add_argument(
  "-p", "--postprocess", 
  help = ("The location of the postprocessing script to run to combine the "
          "results."),
  type = str,
  default = join(dirname(realpath(__file__)), "combine_benchmark_results.py"),
  metavar = "R"
)

ap.add_argument(
  "-r", "--runs", 
  help = ("Run the benchmark suite `R` times.a),"),
  type = int, default = 5, 
  metavar = "R"
)

args = ap.parse_args()

if args.runs <= 0:
  print "ERROR: `--runs` must be greater than `0`."
  ap.print_help()
  exit(1)

BENCHMARK_EXE             = args.benchmark
BENCHMARK_NAME            = basename(BENCHMARK_EXE)
POSTPROCESS_EXE           = args.postprocess
OUTPUT_FILE_NAME          = lambda i: BENCHMARK_NAME + "_" + str(i) + ".csv"
COMBINED_OUTPUT_FILE_NAME = BENCHMARK_NAME + "_combined.csv"

###############################################################################

print '&&&& RUNNING {0}'.format(BENCHMARK_NAME)

print '#### RUNS {0}'.format(args.runs)

###############################################################################

print '#### CMD {0}'.format(BENCHMARK_EXE)

for i in xrange(args.runs):
  with open(OUTPUT_FILE_NAME(i), "w") as output_file:
    print '#### RUN {0} OUTPUT -> {1}'.format(i, OUTPUT_FILE_NAME(i))

    p = None

    try:
      p = Popen(BENCHMARK_EXE, stdout = output_file, stderr = output_file)
      p.communicate()
    except OSError as ex:
      print_file(OUTPUT_FILE_NAME(i))
      print '#### ERROR Caught OSError `{0}`.'.format(ex)
      print '&&&& FAILED {0}'.format(BENCHMARK_NAME)
      exit(-1)

  print_file(OUTPUT_FILE_NAME(i))

  if p.returncode != 0:
    print '#### ERROR Process exited with code {0}.'.format(p.returncode)
    print '&&&& FAILED {0}'.format(BENCHMARK_NAME)
    exit(p.returncode)

###############################################################################

post_cmd = [POSTPROCESS_EXE]

# Add dependent variable options.
post_cmd += ["-dSTL Average Walltime,STL Walltime Uncertainty,STL Trials"]
post_cmd += ["-dSTL Average Throughput,STL Throughput Uncertainty,STL Trials"]
post_cmd += ["-dThrust Average Walltime,Thrust Walltime Uncertainty,Thrust Trials"]
post_cmd += ["-dThrust Average Throughput,Thrust Throughput Uncertainty,Thrust Trials"]

post_cmd += [OUTPUT_FILE_NAME(i) for i in range(args.runs)] 

print '#### CMD {0}'.format(printable_cmd(post_cmd))

with open(COMBINED_OUTPUT_FILE_NAME, "w") as output_file:
  p = None

  try:
    p = Popen(post_cmd, stdout = output_file, stderr = output_file)
    p.communicate()
  except OSError as ex:
    print_file(COMBINED_OUTPUT_FILE_NAME)
    print '#### ERROR Caught OSError `{0}`.'.format(ex)
    print '&&&& FAILED {0}'.format(BENCHMARK_NAME)
    exit(-1)

  print_file(COMBINED_OUTPUT_FILE_NAME)

  if p.returncode != 0:
    print '#### ERROR Process exited with code {0}.'.format(p.returncode)
    print '&&&& FAILED {0}'.format(BENCHMARK_NAME)
    exit(p.returncode)

  with open(COMBINED_OUTPUT_FILE_NAME) as input_file:
    reader = csv_dict_reader(input_file)

    variable_units = reader.next() # Get units header row.

    distinguishing_variables = reader.fieldnames

    measured_variables = [
      ("STL Average Throughput",    "+"),
      ("Thrust Average Throughput", "+")
    ]

    for record in reader:
      for variable, directionality in measured_variables:
        # Don't monitor regressions for STL implementations, nvbug 28980890:
        if "STL" in variable:
          continue
        print "&&&& PERF {0}_{1}_{2}bit_{3}mib_{4} {5} {6}{7}".format(
          record["Algorithm"],
          record["Element Type"],
          record["Element Size"],
          record["Total Input Size"],
          variable.replace(" ", "_").lower(),
          record[variable],
          directionality,
          variable_units[variable]
        )

###############################################################################
                  
print '&&&& PASSED {0}'.format(BENCHMARK_NAME)

