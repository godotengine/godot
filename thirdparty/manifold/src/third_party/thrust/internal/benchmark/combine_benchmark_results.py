#! /usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# Copyright (c) 2012-7 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
###############################################################################

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

# XXX Put code shared with `compare_benchmark_results.py` in a common place.

# XXX Relative uncertainty.

from sys import exit, stdout

from os.path import splitext

from itertools import imap # Lazy map.

from math import sqrt, log10, floor

from collections import deque

from argparse import ArgumentParser as argument_parser

from csv import DictReader as csv_dict_reader
from csv import DictWriter as csv_dict_writer

from re import compile as regex_compile

###############################################################################

def unpack_tuple(f):
  """Return a unary function that calls `f` with its argument unpacked."""
  return lambda args: f(*iter(args))

def strip_dict(d):
  """Strip leading and trailing whitespace from all keys and values in `d`."""
  d.update({key: value.strip() for (key, value) in d.items()})

def merge_dicts(d0, d1):
  """Create a new `dict` that is the union of `dict`s `d0` and `d1`."""
  d = d0.copy()
  d.update(d1)
  return d

def strip_list(l):
  """Strip leading and trailing whitespace from all values in `l`."""
  for i, value in enumerate(l): l[i] = value.strip()

###############################################################################

def int_or_float(x):
  """Convert `x` to either `int` or `float`, preferring `int`.

  Raises:
    ValueError : If `x` is not convertible to either `int` or `float`
  """
  try:
    return int(x)
  except ValueError:
    return float(x)

def try_int_or_float(x):
  """Try to convert `x` to either `int` or `float`, preferring `int`. `x` is
  returned unmodified if conversion fails.
  """
  try:
    return int_or_float(x)
  except ValueError:
    return x

###############################################################################

def find_significant_digit(x):
  """Return the significant digit of the number x. The result is the number of
  digits after the decimal place to round to (negative numbers indicate rounding
  before the decimal place)."""
  if x == 0: return 0
  return -int(floor(log10(abs(x))))

def round_with_int_conversion(x, ndigits = None):
  """Rounds `x` to `ndigits` after the the decimal place. If `ndigits` is less
  than 1, convert the result to `int`. If `ndigits` is `None`, the significant
  digit of `x` is used."""
  if ndigits is None: ndigits = find_significant_digit(x)
  x_rounded = round(x, ndigits)
  return int(x_rounded) if ndigits < 1 else x_rounded

###############################################################################

class measured_variable(object):
  """A meta-variable representing measured data. It is composed of three raw
  variables plus units meta-data.

  Attributes:
    quantity (`str`) :
      Name of the quantity variable of this object.
    uncertainty (`str`) :
      Name of the uncertainty variable of this object.
    sample_size (`str`) :
      Name of the sample size variable of this object.
    units (units class or `None`) :
      The units the value is measured in.
  """

  def __init__(self, quantity, uncertainty, sample_size, units = None):
    self.quantity    = quantity
    self.uncertainty = uncertainty
    self.sample_size = sample_size
    self.units       = units

  def as_tuple(self):
    return (self.quantity, self.uncertainty, self.sample_size, self.units)

  def __iter__(self):
    return iter(self.as_tuple())

  def __str__(self):
    return str(self.as_tuple())

  def __repr__(self):
    return str(self)

class measured_value(object):
  """An object that represents a value determined by multiple measurements.

  Attributes:
    quantity (scalar) :
      The quantity of the value, e.g. the arithmetic mean.
    uncertainty (scalar) :
      The measurement uncertainty, e.g. the sample standard deviation.
    sample_size (`int`) :
      The number of observations contributing to the value.
    units (units class or `None`) :
      The units the value is measured in.
  """

  def __init__(self, quantity, uncertainty, sample_size = 1, units = None):
    self.quantity    = quantity
    self.uncertainty = uncertainty
    self.sample_size = sample_size
    self.units       = units

  def as_tuple(self):
    return (self.quantity, self.uncertainty, self.sample_size, self.units)

  def __iter__(self):
    return iter(self.as_tuple())

  def __str__(self):
    return str(self.as_tuple())

  def __repr__(self):
    return str(self)

###############################################################################

def arithmetic_mean(X):
  """Computes the arithmetic mean of the sequence `X`.

  Let:

    * `n = len(X)`.
    * `u` denote the arithmetic mean of `X`.

  .. math::

    u = \frac{\sum_{i = 0}^{n - 1} X_i}{n}
  """
  return sum(X) / len(X)

def sample_variance(X, u = None):
  """Computes the sample variance of the sequence `X`.

  Let:

    * `n = len(X)`.
    * `u` denote the arithmetic mean of `X`.
    * `s` denote the sample standard deviation of `X`.

  .. math::

    v = \frac{\sum_{i = 0}^{n - 1} (X_i - u)^2}{n - 1}

  Args:
    X (`Iterable`) : The sequence of values.
    u (number)     : The arithmetic mean of `X`.
  """
  if u is None: u = arithmetic_mean(X)
  return sum(imap(lambda X_i: (X_i - u) ** 2, X)) / (len(X) - 1)
 
def sample_standard_deviation(X, u = None, v = None):
  """Computes the sample standard deviation of the sequence `X`.

  Let:

    * `n = len(X)`.
    * `u` denote the arithmetic mean of `X`.
    * `v` denote the sample variance of `X`.
    * `s` denote the sample standard deviation of `X`.

  .. math::

    s &= \sqrt{v}
      &= \sqrt{\frac{\sum_{i = 0}^{n - 1} (X_i - u)^2}{n - 1}}

  Args:
    X (`Iterable`) : The sequence of values.
    u (number)     : The arithmetic mean of `X`.
    v (number)     : The sample variance of `X`.
  """
  if u is None: u = arithmetic_mean(X)
  if v is None: v = sample_variance(X, u)
  return sqrt(v)

def combine_sample_size(As):
  """Computes the combined sample variance of a group of `measured_value`s.

  Let:

    * `g = len(As)`.
    * `n_i = As[i].samples`.
    * `n` denote the combined sample size of `As`.

  .. math::

    n = \sum{i = 0}^{g - 1} n_i
  """
  return sum(imap(unpack_tuple(lambda u_i, s_i, n_i, t_i: n_i), As))

def combine_arithmetic_mean(As, n = None):
  """Computes the combined arithmetic mean of a group of `measured_value`s.

  Let:

    * `g = len(As)`.
    * `u_i = As[i].quantity`.
    * `n_i = As[i].samples`.
    * `n` denote the combined sample size of `As`.
    * `u` denote the arithmetic mean of the quantities of `As`.

  .. math::

    u = \frac{\sum{i = 0}^{g - 1} n_i u_i}{n}
  """
  if n is None: n = combine_sample_size(As)
  return sum(imap(unpack_tuple(lambda u_i, s_i, n_i, t_i: n_i * u_i), As)) / n
  
def combine_sample_variance(As, n = None, u = None):
  """Computes the combined sample variance of a group of `measured_value`s.

  Let:

    * `g = len(As)`.
    * `u_i = As[i].quantity`.
    * `s_i = As[i].uncertainty`.
    * `n_i = As[i].samples`.
    * `n` denote the combined sample size of `As`.
    * `u` denote the arithmetic mean of the quantities of `As`.
    * `v` denote the sample variance of `X`.

  .. math::

    v = \frac{(\sum_{i = 0}^{g - 1} n_i (u_i - u)^2 + s_i^2 (n_i - 1))}{n - 1}

  Args:
    As (`Iterable` of `measured_value`s) : The sequence of values.
    n (number)                           : The combined sample sizes of `As`.
    u (number)                           : The combined arithmetic mean of `As`.
  """
  if n <= 1: return 0
  if n is None: n = combine_sample_size(As)
  if u is None: u = combine_arithmetic_mean(As, n)
  return sum(imap(unpack_tuple(
    lambda u_i, s_i, n_i, t_i: n_i * (u_i - u) ** 2 + (s_i ** 2) * (n_i - 1)
  ), As)) / (n - 1)

def combine_sample_standard_deviation(As, n = None, u = None, v = None):
  """Computes the combined sample standard deviation of a group of
  `measured_value`s.

  Let:

    * `g = len(As)`.
    * `u_i = As[i].quantity`.
    * `s_i = As[i].uncertainty`.
    * `n_i = As[i].samples`.
    * `n` denote the combined sample size of `As`.
    * `u` denote the arithmetic mean of the quantities of `As`.
    * `v` denote the sample variance of `X`.
    * `s` denote the sample standard deviation of `X`.

  .. math::

    s &= \sqrt{v}
      &= \sqrt{\frac{(\sum_{i = 0}^{g - 1} n_i (u_i - u)^2 + s_i^2 (n_i - 1))}{n - 1}}

  Args:
    As (`Iterable` of `measured_value`s) : The sequence of values.
    n (number)                           : The combined sample sizes of `As`.
    u (number)                           : The combined arithmetic mean of `As`.
    v (number)                           : The combined sample variance of `As`.
  """
  if n <= 1: return 0
  if n is None: n = combine_sample_size(As)
  if u is None: u = combine_arithmetic_mean(As, n)
  if v is None: v = combine_sample_variance(As, n, u)
  return sqrt(v)

###############################################################################

def process_program_arguments():
  ap = argument_parser(
    description = (
      "Aggregates the results of multiple runs of benchmark results stored in "
      "CSV format."
    )
  )

  ap.add_argument(
    "-d", "--dependent-variable",
    help = ("Treat the specified three variables as a dependent variable. The "
            "1st variable is the measured quantity, the 2nd is the uncertainty "
            "of the measurement and the 3rd is the sample size. The defaults "
            "are the dependent variables of Thrust's benchmark suite. May be "
            "specified multiple times."),
    action = "append", type = str, dest = "dependent_variables",
    metavar = "QUANTITY,UNCERTAINTY,SAMPLES"
  )

  ap.add_argument(
    "-p", "--preserve-whitespace",
    help = ("Don't trim leading and trailing whitespace from each CSV cell."),
    action = "store_true", default = False
  )

  ap.add_argument(
    "-o", "--output-file",
    help = ("The file that results are written to. If `-`, results are "
            "written to stdout."),
    action = "store", type = str, default = "-",
    metavar = "OUTPUT"
  )

  ap.add_argument(
    "input_files",
    help = ("Input CSV files. The first two rows should be a header. The 1st "
            "header row specifies the name of each variable, and the 2nd "
            "header row specifies the units for that variable."),
    type = str, nargs = "+",
    metavar = "INPUTS"
  )

  return ap.parse_args()

###############################################################################

def filter_comments(f, s = "#"):
  """Return an iterator to the file `f` which filters out all lines beginning
  with `s`."""
  return filter(lambda line: not line.startswith(s), f)

###############################################################################

class io_manager(object):
  """Manages I/O operations and represents the input data as an `Iterable`
  sequence of `dict`s.

  It is `Iterable` and an `Iterator`. It can be used with `with`.

  Attributes:
    preserve_whitespace (`bool`) :
      If `False`, leading and trailing whitespace is stripped from each CSV cell.
    writer (`csv_dict_writer`) :
      CSV writer object that the output is written to.
    output_file (`file` or `stdout`) :
      The output `file` object.
    readers (`list` of `csv_dict_reader`s) :
      List of input files as CSV reader objects.
    input_files (list of `file`s) :
      List of input `file` objects.
    variable_names (`list` of `str`s) :
      Names of the variables, in order. 
    variable_units (`list` of `str`s) :
      Units of the variables, in order. 
  """

  def __init__(self, input_files, output_file, preserve_whitespace = True):
    """Read input files and open the output file and construct a new `io_manager`
    object.

    If `preserve_whitespace` is `False`, leading and trailing whitespace is
    stripped from each CSV cell.

    Raises
      AssertionError :
        If `len(input_files) <= 0` or `type(preserve_whitespace) != bool`.
    """
    assert len(input_files) > 0, "No input files provided."

    assert type(preserve_whitespace) == bool

    self.preserve_whitespace = preserve_whitespace

    self.readers = deque()

    self.variable_names = None
    self.variable_units = None

    self.input_files = deque()

    for input_file in input_files:
      input_file_object = open(input_file)
      reader = csv_dict_reader(filter_comments(input_file_object))

      if not self.preserve_whitespace:
        strip_list(reader.fieldnames)

      if self.variable_names is None:
        self.variable_names = reader.fieldnames
      else:
        # Make sure all inputs have the same schema.
        assert self.variable_names == reader.fieldnames,                      \
          "Input file (`" + input_file + "`) variable schema `"             + \
          str(reader.fieldnames) + "` does not match the variable schema `" + \
          str(self.variable_names) + "`."

      # Consume the next row, which should be the second line of the header.
      variable_units = reader.next()

      if not self.preserve_whitespace:
        strip_dict(variable_units)

      if self.variable_units is None:
        self.variable_units = variable_units
      else:
        # Make sure all inputs have the same units schema.
        assert self.variable_units == variable_units,                         \
          "Input file (`" + input_file + "`) units schema `"                + \
          str(variable_units) + "` does not match the units schema `"       + \
          str(self.variable_units) + "`."

      self.readers.append(reader)
      self.input_files.append(input_file_object)
 
    if   output_file == "-": # Output to stdout.
      self.output_file = stdout
    else:                    # Output to user-specified file.
      self.output_file = open(output_file, "w")

    self.writer = csv_dict_writer(
      self.output_file, fieldnames = self.variable_names
    )

  def __enter__(self):
    """Called upon entering a `with` statement."""
    return self

  def __exit__(self, *args):
    """Called upon exiting a `with` statement."""
    if   self.output_file is stdout:
      self.output_file = None
    elif self.output_file is not None:
      self.output_file.__exit__(*args)

    for input_file in self.input_files:
      input_file.__exit__(*args)

  #############################################################################
  # Input Stream.

  def __iter__(self):
    """Return an iterator to the input sequence.

    This is a requirement for the `Iterable` protocol.
    """
    return self

  def next(self):
    """Consume and return the next record (a `dict` representing a CSV row) in
    the input.

    This is a requirement for the `Iterator` protocol.

    Raises:
      StopIteration : If there is no more input.
    """
    if len(self.readers) == 0:
      raise StopIteration()

    try:
      row = self.readers[0].next()
      if not self.preserve_whitespace: strip_dict(row)
      return row
    except StopIteration:
      # The current reader is empty, so pop it, pop it's input file, close the
      # input file, and then call ourselves again. 
      self.readers.popleft()
      self.input_files.popleft().close()
      return self.next()

  #############################################################################
  # Output.

  def write_header(self):
    """Write the header for the output CSV file."""
    # Write the first line of the header.
    self.writer.writeheader()

    # Write the second line of the header.
    self.writer.writerow(self.variable_units)

  def write(self, d):
    """Write a record (a `dict`) to the output CSV file."""
    self.writer.writerow(d)

###############################################################################

class dependent_variable_parser(object):
  """Parses a `--dependent-variable=AVG,STDEV,TRIALS` command line argument."""

  #############################################################################
  # Grammar

  # Parse a variable_name.
  variable_name_rule = r'[^,]+'

  # Parse a variable classification.        
  dependent_variable_rule = r'(' + variable_name_rule + r')'   \
                          + r','                               \
                          + r'(' + variable_name_rule + r')'   \
                          + r','                               \
                          + r'(' + variable_name_rule + r')'

  engine = regex_compile(dependent_variable_rule)

  #############################################################################

  def __call__(self, s):
    """Parses the string `s` with the form "AVG,STDEV,TRIALS".

    Returns:
      A `measured_variable`. 

    Raises:
      AssertionError : If parsing fails.
    """

    match = self.engine.match(s)

    assert match is not None,                                          \
      "Dependent variable (-d) `" +s+ "` is invalid, the format is " + \
      "`AVG,STDEV,TRIALS`."

    return measured_variable(match.group(1), match.group(2), match.group(3))

###############################################################################

class record_aggregator(object):
  """Consumes and combines records and represents the result as an `Iterable`
  sequence of `dict`s.

  It is `Iterable` and an `Iterator`.

  Attributes:
    dependent_variables (`list` of `measured_variable`s) :
      A list of dependent variables provided on the command line.
    dataset (`dict`) :
      A mapping of distinguishing (e.g. control + independent) values (`tuple`s
      of variable-quantity pairs) to `list`s of dependent values (`dict`s from 
      variables to lists of cells).
    in_order_dataset_keys :
      A list of unique dataset keys (e.g. distinguishing variables) in order of
      appearance.
  """

  parse_dependent_variable = dependent_variable_parser()

  def __init__(self, raw_dependent_variables):
    """Parse dependent variables and construct a new `record_aggregator` object.

    Raises:
      AssertionError : If parsing of dependent variables fails.
    """
    self.dependent_variables = []

    if raw_dependent_variables is not None:
      for variable in raw_dependent_variables:
        self.dependent_variables.append(self.parse_dependent_variable(variable))

    self.dataset = {}

    self.in_order_dataset_keys = deque()

  #############################################################################
  # Insertion.

  def append(self, record):
    """Add `record` to the dataset.

    Raises:
      ValueError : If any `str`-to-numeric conversions fail.
    """
    # The distinguishing variables are the control and independent variables.
    # They form the key for each record in the dataset. Records with the same
    # distinguishing variables are treated as observations of the same data
    # point.
    dependent_values = {}

    # To allow the same sample size variable to be used for multiple dependent
    # variables, we don't pop sample size variables until we're done processing
    # all variables.
    sample_size_variables = []

    # Separate the dependent values from the distinguishing variables and
    # perform `str`-to-numeric conversions.
    for variable in self.dependent_variables:
      quantity, uncertainty, sample_size, units = variable.as_tuple()

      dependent_values[quantity]    = [int_or_float(record.pop(quantity))]
      dependent_values[uncertainty] = [int_or_float(record.pop(uncertainty))]
      dependent_values[sample_size] = [int(record[sample_size])]

      sample_size_variables.append(sample_size)

    # Pop sample size variables.
    for sample_size_variable in sample_size_variables:
      # Allowed to fail, as we may have duplicates.
      record.pop(sample_size_variable, None)

    # `dict`s aren't hashable, so create a tuple of key-value pairs.
    distinguishing_values = tuple(record.items())

    if distinguishing_values in self.dataset:
      # These distinguishing values already exist, so get the `dict` they're
      # mapped to, look up each key in `dependent_values` in the `dict`, and
      # add the corresponding quantity in `dependent_values` to the list in the
      # the `dict`.
      for variable, columns in dependent_values.iteritems():
        self.dataset[distinguishing_values][variable] += columns
    else:
      # These distinguishing values aren't in the dataset, so add them and
      # record them in `in_order_dataset_keys`.
      self.dataset[distinguishing_values] = dependent_values
      self.in_order_dataset_keys.append(distinguishing_values)

  #############################################################################
  # Postprocessing.

  def combine_dependent_values(self, dependent_values):
    """Takes a mapping of dependent variables to lists of cells and returns
    a new mapping with the cells combined.

    Raises:
      AssertionError : If class invariants were violated.
    """
    combined_dependent_values = dependent_values.copy()

    for variable in self.dependent_variables:
      quantity, uncertainty, sample_size, units = variable.as_tuple()

      quantities    = dependent_values[quantity]
      uncertainties = dependent_values[uncertainty]
      sample_sizes  = dependent_values[sample_size]

      if type(sample_size) is list:
        # Sample size hasn't been combined yet.
        assert len(quantities)    == len(uncertainties)                       \
           and len(uncertainties) == len(sample_sizes),                       \
          "Length of quantities list `(" + str(len(quantities)) + ")`, "    + \
          "length of uncertainties list `(" + str(len(uncertainties))       + \
          "),` and length of sample sizes list `(" + str(len(sample_sizes)) + \
          ")` are not the same."
      else:
        # Another dependent variable that uses our sample size has combined it
        # already.
        assert len(quantities) == len(uncertainties),                         \
          "Length of quantities list `(" + str(len(quantities)) + ")` and " + \
          "length of uncertainties list `(" + str(len(uncertainties))       + \
          ")` are not the same."

      # Convert the three separate `list`s into one list of `measured_value`s.
      measured_values = []

      for i in range(len(quantities)):
        mv = measured_value(
          quantities[i], uncertainties[i], sample_sizes[i], units
        )

        measured_values.append(mv)

      # Combine the `measured_value`s.
      combined_sample_size = combine_sample_size(
        measured_values
      )

      combined_arithmetic_mean = combine_arithmetic_mean(
        measured_values, combined_sample_size
      )

      combined_sample_standard_deviation = combine_sample_standard_deviation(
        measured_values, combined_sample_size, combined_arithmetic_mean
      )

      # Round the quantity and uncertainty to the significant digit of
      # uncertainty and insert the combined values into the results.
      sigdig = find_significant_digit(combined_sample_standard_deviation)

#      combined_arithmetic_mean = round_with_int_conversion(
#        combined_arithmetic_mean, sigdig
#      )

#      combined_sample_standard_deviation = round_with_int_conversion(
#        combined_sample_standard_deviation, sigdig
#      )

      combined_dependent_values[quantity]    = combined_arithmetic_mean
      combined_dependent_values[uncertainty] = combined_sample_standard_deviation
      combined_dependent_values[sample_size] = combined_sample_size

    return combined_dependent_values

  ############################################################################# 
  # Output Stream.

  def __iter__(self):
    """Return an iterator to the output sequence of separated distinguishing
    variables and dependent variables (a tuple of two `dict`s).

    This is a requirement for the `Iterable` protocol.
    """
    return self

  def records(self):
    """Return an iterator to the output sequence of CSV rows (`dict`s of
    variables to values).
    """
    return imap(unpack_tuple(lambda dist, dep: merge_dicts(dist, dep)), self)

  def next(self):
    """Produce the components of the next output record - a tuple of two
    `dict`s. The first `dict` is a mapping of distinguishing variables to
    distinguishing values, the second `dict` is a mapping of dependent
    variables to combined dependent values. Combining the two dicts forms a
    CSV row suitable for output.

    This is a requirement for the `Iterator` protocol.

    Raises:
      StopIteration  : If there is no more output.
      AssertionError : If class invariants were violated.
    """
    assert len(self.dataset.keys()) == len(self.in_order_dataset_keys),      \
      "Number of dataset keys (`" + str(len(self.dataset.keys()))          + \
      "`) is not equal to the number of keys in the ordering list (`"      + \
      str(len(self.in_order_dataset_keys)) + "`)."

    if len(self.in_order_dataset_keys) == 0:
      raise StopIteration()

    # Get the next set of distinguishing values and convert them to a `dict`.
    raw_distinguishing_values = self.in_order_dataset_keys.popleft()
    distinguishing_values     = dict(raw_distinguishing_values)

    dependent_values = self.dataset.pop(raw_distinguishing_values)

    combined_dependent_values = self.combine_dependent_values(dependent_values)

    return (distinguishing_values, combined_dependent_values)

###############################################################################

args = process_program_arguments()

if args.dependent_variables is None:
  args.dependent_variables = [
    "STL Average Walltime,STL Walltime Uncertainty,STL Trials",
    "STL Average Throughput,STL Throughput Uncertainty,STL Trials",
    "Thrust Average Walltime,Thrust Walltime Uncertainty,Thrust Trials",
    "Thrust Average Throughput,Thrust Throughput Uncertainty,Thrust Trials"
  ]

# Read input files and open the output file.
with io_manager(args.input_files,
                args.output_file,
                args.preserve_whitespace) as iom:
  # Parse dependent variable options.
  ra = record_aggregator(args.dependent_variables)

  # Add all input data to the `record_aggregator`.
  for record in iom:
    ra.append(record)

  iom.write_header()

  # Write combined results out.
  for record in ra.records():
    iom.write(record)

