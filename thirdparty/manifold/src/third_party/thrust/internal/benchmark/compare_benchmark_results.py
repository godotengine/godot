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

# XXX Put code shared with `combine_benchmark_results.py` in a common place.

# XXX Relative uncertainty.

# XXX Create uncertain value class which is quantity + uncertainty.

from sys import exit, stdout

from os.path import splitext

from itertools import imap # Lazy map.

from math import sqrt, log10, floor

from collections import deque

from argparse import ArgumentParser as argument_parser
from argparse import Action as argument_action

from csv import DictReader as csv_dict_reader
from csv import DictWriter as csv_dict_writer

from re import compile as regex_compile

###############################################################################

def unpack_tuple(f):
  """Return a unary function that calls `f` with its argument unpacked."""
  return lambda args: f(*iter(args))

def strip_dict(d):
  """Strip leading and trailing whitespace from all keys and values in `d`.

  Returns:
    The modified dict `d`.
  """
  d.update({key: value.strip() for (key, value) in d.items()})
  return d

def merge_dicts(d0, d1):
  """Create a new `dict` that is the union of `dict`s `d0` and `d1`."""
  d = d0.copy()
  d.update(d1)
  return d

def change_key_in_dict(d, old_key, new_key):
  """Change the key of the entry in `d` with key `old_key` to `new_key`. If
  there is an existing entry 

  Returns:
    The modified dict `d`.

  Raises:
    KeyError : If `old_key` is not in `d`.
  """
  d[new_key] = d.pop(old_key)
  return d

def key_from_dict(d):
  """Create a hashable key from a `dict` by converting the `dict` to a tuple."""
  return tuple(sorted(d.items()))

def strip_list(l):
  """Strip leading and trailing whitespace from all values in `l`."""
  for i, value in enumerate(l): l[i] = value.strip()
  return l

def remove_from_list(l, item):
  """Remove the first occurence of `item` from list `l` and return a tuple of
  the index that was removed and the element that was removed.

  Raises:
    ValueError : If `item` is not in `l`.
  """
  idx = l.index(item)
  item = l.pop(idx)
  return (idx, item)

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

def ranges_overlap(x1, x2, y1, y2):
  """Returns true if the ranges `[x1, x2]` and `[y1, y2]` overlap,
  where `x1 <= x2` and `y1 <= y2`.

  Raises:
    AssertionError : If `x1 > x2` or `y1 > y2`.
  """
  assert x1 <= x2
  assert y1 <= y2
  return x1 <= y2 and y1 <= x2

def ranges_overlap_uncertainty(x, x_unc, y, y_unc):
  """Returns true if the ranges `[x - x_unc, x + x_unc]` and
  `[y - y_unc, y + y_unc]` overlap, where `x_unc >= 0` and `y_unc >= 0`.

  Raises:
    AssertionError : If `x_unc < 0` or `y_unc < 0`.
  """
  assert x_unc >= 0
  assert y_unc >= 0
  return ranges_overlap(x - x_unc, x + x_unc, y - y_unc, y + y_unc)

###############################################################################

# Formulas for propagation of uncertainty from:
#
#   https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulas
#
# Even though it's Wikipedia, I trust it as I helped write that table.
#
# XXX Replace with a proper reference.

def uncertainty_multiplicative(f, A, A_abs_unc, B, B_abs_unc):
  """Compute the propagated uncertainty from the multiplication of two
  uncertain values, `A +/- A_abs_unc` and `B +/- B_abs_unc`. Given `f = AB` or
  `f = A/B`, where `A != 0` and `B != 0`, the uncertainty in `f` is
  approximately:

  .. math::

    \sigma_f = |f| \sqrt{\frac{\sigma_A}{A} ^ 2 + \frac{\sigma_B}{B} ^ 2}

  Raises:
    ZeroDivisionError : If `A == 0` or `B == 0`.
  """
  return abs(f) * sqrt((A_abs_unc / A) ** 2 + (B_abs_unc / B) ** 2);

def uncertainty_additive(c, A_abs_unc, d, B_abs_unc):
  """Compute the propagated uncertainty from addition of two uncertain values,
  `A +/- A_abs_unc` and `B +/- B_abs_unc`. Given `f = cA + dB`, where `c` and
  `d` are certain constants, the uncertainty in `f` is approximately:

  .. math::

    f_{\sigma} = \sqrt{c ^ 2 * A_{\sigma} ^ 2 + d ^ 2 * B_{\sigma} ^ 2}
  """
  return sqrt(((c ** 2) * (A_abs_unc ** 2)) + ((d ** 2) * (B_abs_unc ** 2)))

###############################################################################

# XXX Create change class.

def absolute_change(old, new):
  """Computes the absolute change from old to new:

  .. math::

    absolute_change = new - old
  """
  return new - old

def absolute_change_uncertainty(old, old_unc, new, new_unc):
  """Computes the uncertainty in the absolute change from old to new and returns
  a tuple of the absolute change and the absolute change uncertainty.
  """
  absolute_change     = new - old
  absolute_change_unc = uncertainty_additive(1.0, new_unc, -1.0, old_unc)

  return (absolute_change, absolute_change_unc)

def percent_change(old, new):
  """Computes the percent change from old to new:

  .. math::

    percent_change = 100 \frac{new - old}{abs(old)}
  """
  return float(new - old) / abs(old)

def percent_change_uncertainty(old, old_unc, new, new_unc):
  """Computes the uncertainty in the percent change from old to new and returns
  a tuple of the absolute change, the absolute change uncertainty, the percent
  change and the percent change uncertainty.
  """
  # Let's break this down into a few sub-operations:
  # 
  #   absolute_change = new - old         <- Additive propagation.
  #   relative_change = change / abs(old) <- Multiplicative propagation.
  #   percent_change  = 100 * y           <- Multiplicative propagation.

  if old == 0:
    # We can't compute relative change because the old value is 0.
    return (float("nan"), float("nan"), float("nan"), float("nan"))

  (absolute_change, absolute_change_unc) = absolute_change_uncertainty(
    old, old_unc, new, new_unc
  )

  if absolute_change == 0:
    # We can't compute relative change uncertainty because the relative
    # uncertainty of a value of 0 is undefined.
    return (absolute_change, absolute_change_unc, float("nan"), float("nan"))

  relative_change     = float(absolute_change) / abs(old)
  relative_change_unc = uncertainty_multiplicative(
    relative_change, absolute_change, absolute_change_unc, old, old_unc
  )

  percent_change = 100.0 * relative_change
  percent_change_unc = uncertainty_multiplicative(
    percent_change, 100.0, 0.0, relative_change, relative_change_unc
  )

  return (
    absolute_change, absolute_change_unc, percent_change, percent_change_unc
  )

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
    v &= \frac{(\sum_{i = 0}^{g - 1} n_i (u_i - u)^2 + s_i^2 (n_i - 1))}{n - 1}

    s &= \sqrt{v}

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

def store_const_multiple(const, *destinations):
  """Returns an `argument_action` class that sets multiple argument
  destinations (`destinations`) to `const`."""
  class store_const_multiple_action(argument_action):
    def __init__(self, *args, **kwargs):
      super(store_const_multiple_action, self).__init__(
        metavar = None, nargs = 0, const = const, *args, **kwargs
      )

    def __call__(self, parser, namespace, values, option_string = None):
      for destination in destinations:
        setattr(namespace, destination, const)

  return store_const_multiple_action

def store_true_multiple(*destinations):
  """Returns an `argument_action` class that sets multiple argument
  destinations (`destinations`) to `True`."""
  return store_const_multiple(True, *destinations)

def store_false_multiple(*destinations):
  """Returns an `argument_action` class that sets multiple argument
  destinations (`destinations`) to `False`."""
  return store_const_multiple(False, *destinations)

###############################################################################

def process_program_arguments():
  ap = argument_parser(
    description = (
      "Compares two sets of combined performance results and identifies "
      "statistically significant changes."
    )
  )

  ap.add_argument(
    "baseline_input_file",
    help = ("CSV file containing the baseline performance results. The first "
            "two rows should be a header. The 1st header row specifies the "
            "name of each variable, and the 2nd header row specifies the units "
            "for that variable. The baseline results may be a superset of the "
            "observed performance results, but the reverse is not true. The "
            "baseline results must contain data for every datapoint in the "
            "observed performance results."),            
    type = str
  )

  ap.add_argument(
    "observed_input_file",
    help = ("CSV file containing the observed performance results. The first "
            "two rows should be a header. The 1st header row specifies the name "
            "of header row specifies the units for that variable."),
    type = str
  )

  ap.add_argument(
    "-o", "--output-file",
    help = ("The file that results are written to. If `-`, results are "
            "written to stdout."),
    action = "store", type = str, default = "-",
    metavar = "OUTPUT"
  )

  ap.add_argument(
    "-c", "--control-variable",
    help = ("Treat the specified variable as a control variable. This means "
            "it will be filtered out when forming dataset keys. For example, "
            "this could be used to ignore a timestamp variable that is "
            "different in the baseline and observed results. May be specified "
            "multiple times."),
    action = "append", type = str, dest = "control_variables", default = [],
    metavar = "QUANTITY"
  )

  ap.add_argument(
    "-d", "--dependent-variable",
    help = ("Treat the specified three variables as a dependent variable. The "
            "1st variable is the measured quantity, the 2nd is the uncertainty "
            "of the measurement and the 3rd is the sample size. The defaults "
            "are the dependent variables of Thrust's benchmark suite. May be "
            "specified multiple times."),
    action = "append", type = str, dest = "dependent_variables", default = [],
    metavar = "QUANTITY,UNCERTAINTY,SAMPLES"
  )

  ap.add_argument(
    "-t", "--change-threshold",
    help = ("Treat relative changes less than this amount (a percentage) as "
            "statistically insignificant. The default is 5%%."),
    action = "store", type = float, default = 5,
    metavar = "PERCENTAGE"
  )

  ap.add_argument(
    "-p", "--preserve-whitespace",
    help = ("Don't trim leading and trailing whitespace from each CSV cell."),
    action = "store_true", default = False
  )

  ap.add_argument(
    "--output-all-variables",
    help = ("Don't omit original absolute values in output."),
    action = "store_true", default = False
  )

  ap.add_argument(
    "--output-all-datapoints",
    help = ("Don't omit datapoints that are statistically indistinguishable "
            "in output."),
    action = "store_true", default = False
  )

  ap.add_argument(
    "-a", "--output-all",
    help = ("Equivalent to `--output-all-variables --output-all-datapoints`."),
    action = store_true_multiple("output_all_variables", "output_all_datapoints")
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
    baseline_reader (`csv_dict_reader`) :
      CSV reader object for the baseline results.
    observed_reader (`csv_dict_reader`) :
      CSV reader object for the observed results.
    baseline_input_file (`file`) :
      `file` object for the baseline results.
    observed_input_file (`file`) :
      `file` object for the observed results..
    variable_names (`list` of `str`s) :
      Names of the variables, in order. 
    variable_units (`list` of `str`s) :
      Units of the variables, in order. 
  """

  def __init__(self,
               baseline_input_file, observed_input_file,
               output_file,
               preserve_whitespace = False):
    """Read input files and open the output file and construct a new `io_manager`
    object.

    If `preserve_whitespace` is `False`, leading and trailing whitespace is
    stripped from each CSV cell.

    Raises
      AssertionError :
        If `type(preserve_whitespace) != bool`.
    """
    assert type(preserve_whitespace) == bool

    self.preserve_whitespace = preserve_whitespace

    # Open baseline results.
    self.baseline_input_file = open(baseline_input_file)
    self.baseline_reader = csv_dict_reader(
      filter_comments(self.baseline_input_file)
    )

    if not self.preserve_whitespace:
      strip_list(self.baseline_reader.fieldnames)

    self.variable_names = list(self.baseline_reader.fieldnames) # Copy.
    self.variable_units = self.baseline_reader.next()

    if not self.preserve_whitespace:
      strip_dict(self.variable_units)

    # Open observed results.
    self.observed_input_file = open(observed_input_file)
    self.observed_reader = csv_dict_reader(
      filter_comments(self.observed_input_file)
    )

    if not self.preserve_whitespace:
      strip_list(self.observed_reader.fieldnames)

    # Make sure all inputs have the same variables schema.
    assert self.variable_names == self.observed_reader.fieldnames,             \
      "Observed results input file (`" + observed_input_file + "`) "         + \
      "variable schema `" + str(self.observed_reader.fieldnames) + "` does " + \
      "not match the baseline results input file (`" + baseline_input_file   + \
      "`) variable schema `" + str(self.variable_names) + "`."

    # Consume the next row, which should be the second line of the header.
    observed_variable_units = self.observed_reader.next()

    if not self.preserve_whitespace:
      strip_dict(observed_variable_units)

    # Make sure all inputs have the same units schema.
    assert self.variable_units == observed_variable_units,                    \
      "Observed results input file (`" + observed_input_file + "`) "        + \
      "units schema `" + str(observed_variable_units) + "` does not "       + \
      "match the baseline results input file (`" + baseline_input_file      + \
      "`) units schema `" + str(self.variable_units) + "`."

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

    self.baseline_input_file.__exit__(*args)
    self.observed_input_file.__exit__(*args)

  def append_variable(self, name, units):
    """Add a new variable to the output schema."""
    self.variable_names.append(name)
    self.variable_units.update({name : units})

    # Update CSV writer field names.
    self.writer.fieldnames = self.variable_names

  def insert_variable(self, idx, name, units):
    """Insert a new variable into the output schema at index `idx`."""
    self.variable_names.insert(idx, name)
    self.variable_units.update({name : units})

    # Update CSV writer field names.
    self.writer.fieldnames = self.variable_names

  def remove_variable(self, name):
    """Remove variable from the output schema and return a tuple of the variable
    index and the variable units.

    Raises:
      ValueError : If `name` is not in the output schema.
    """
    # Remove the variable and get its index, which we'll need to remove the
    # corresponding units entry.
    (idx, item) = remove_from_list(self.variable_names, name)

    # Remove the units entry.
    units = self.variable_units.pop(item)

    # Update CSV writer field names.
    self.writer.fieldnames = self.variable_names

    return (idx, units)

  #############################################################################
  # Input Stream.

  def baseline(self):
    """Return an iterator to the baseline results input sequence."""
    return imap(lambda row: strip_dict(row), self.baseline_reader) 

  def observed(self):
    """Return an iterator to the observed results input sequence."""
    return imap(lambda row: strip_dict(row), self.observed_reader) 

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
    control_variables (`list` of `str`s) :
      A list of control variables provided on the command line.
    dataset (`dict`) :
      A mapping of distinguishing (e.g. control + independent) values (`tuple`s
      of variable-quantity pairs) to `list`s of dependent values (`dict`s from 
      variables to lists of cells).
    in_order_dataset_keys :
      A list of unique dataset keys (e.g. distinguishing variables) in order of
      appearance.
  """

  def __init__(self, dependent_variables, control_variables):
    """Construct a new `record_aggregator` object.

    Raises:
      AssertionError : If parsing of dependent variables fails.
    """
    self.dependent_variables = dependent_variables
    self.control_variables = control_variables

    self.dataset = {}

    self.in_order_dataset_keys = deque()

  #############################################################################
  # Insertion.

  def key_from_dict(self, d):
    """Create a hashable key from a `dict` by filtering out control variables
    and then converting the `dict` to a tuple.

    Raises:
      AssertionError : If any control variable was not found in `d`.
    """
    distinguishing_values = d.copy()

    # Filter out control variables.
    for var in self.control_variables:
      distinguishing_values.pop(var, None)

    return key_from_dict(distinguishing_values)

  def append(self, record):
    """Add `record` to the dataset.

    Raises:
      ValueError : If any `str`-to-numeric conversions fail.
    """
    # The distinguishing variables are the control and independent variables.
    # They form the key for each record in the dataset. Records with the same
    # distinguishing variables are treated as observations of the same
    # datapoint.
    dependent_values = {}

    # To allow the same sample size variable to be used for multiple dependent
    # variables, we don't pop sample size variables until we're done processing
    # all variables.
    sample_size_variables = []

    # Separate the dependent values from the distinguishing variables and
    # perform `str`-to-numeric conversions.
    for var in self.dependent_variables:
      quantity, uncertainty, sample_size, units = var.as_tuple()

      dependent_values[quantity]    = [int_or_float(record.pop(quantity))]
      dependent_values[uncertainty] = [int_or_float(record.pop(uncertainty))]
      dependent_values[sample_size] = [int(record[sample_size])]

      sample_size_variables.append(sample_size)

    # Pop sample size variables.
    for var in sample_size_variables:
      # Allowed to fail, as we may have duplicates.
      record.pop(var, None)

    distinguishing_values = self.key_from_dict(record)

    if distinguishing_values in self.dataset:
      # These distinguishing values already exist, so get the `dict` they're
      # mapped to, look up each key in `dependent_values` in the `dict`, and
      # add the corresponding quantity in `dependent_values` to the list in the
      # the `dict`.
      for var, columns in dependent_values.iteritems():
        self.dataset[distinguishing_values][var] += columns
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

    for var in self.dependent_variables:
      quantity, uncertainty, sample_size, units = var.as_tuple()

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

  def __getitem__(self, distinguishing_values):
    """Produce the dependent component, a `dict` mapping dependent variables to
    combined dependent values, associated with `distinguishing_values`.

    Args:
      distinguishing_values (`dict`) :
        A `dict` mapping distinguishing variables to distinguishing values.

    Raises:
      KeyError : If `distinguishing_values` is not in the dataset.
    """
    raw_distinguishing_values = self.key_from_dict(distinguishing_values)

    dependent_values = self.dataset[raw_distinguishing_values]

    combined_dependent_values = self.combine_dependent_values(dependent_values)

    return combined_dependent_values

###############################################################################

args = process_program_arguments()

if len(args.dependent_variables) == 0:
  args.dependent_variables = [
    "STL Average Walltime,STL Walltime Uncertainty,STL Trials",
    "STL Average Throughput,STL Throughput Uncertainty,STL Trials",
    "Thrust Average Walltime,Thrust Walltime Uncertainty,Thrust Trials",
    "Thrust Average Throughput,Thrust Throughput Uncertainty,Thrust Trials"
  ]

# Parse dependent variable options.
dependent_variables = []

parse_dependent_variable = dependent_variable_parser()

#if args.dependent_variables is not None:
for var in args.dependent_variables:
  dependent_variables.append(parse_dependent_variable(var))

# Read input files and open the output file.
with io_manager(args.baseline_input_file, 
                args.observed_input_file,
                args.output_file,
                args.preserve_whitespace) as iom:

  # Create record aggregators.
  baseline_ra = record_aggregator(dependent_variables, args.control_variables)
  observed_ra = record_aggregator(dependent_variables, args.control_variables)

  # Duplicate dependent variables: one for baseline results, one for observed
  # results.
  baseline_suffix = " - `{0}`".format(
    args.baseline_input_file
  )
  observed_suffix = " - `{0}`".format(
    args.observed_input_file
  )

  for var in dependent_variables:
    # Remove the existing quantity variable:
    #
    #   [ ..., a, b, c, ... ]
    #             ^- remove b at index i
    #
    (quantity_idx, quantity_units) = iom.remove_variable(var.quantity)

    # If the `--output-all-variables` option was specified, add the new baseline
    # and observed quantity variables. Note that we insert in the reverse of
    # the order we desire (which is baseline then observed):
    #
    #   [ ..., a, b_1, c, ... ]
    #              ^- insert b_1 at index i
    #
    #   [ ..., a, b_0, b_1, c, ... ]
    #              ^- insert b_0 at index i
    #
    if args.output_all_variables:
      iom.insert_variable(
        quantity_idx, var.quantity + observed_suffix, quantity_units
      )
      iom.insert_variable(
        quantity_idx, var.quantity + baseline_suffix, quantity_units
      )

    # Remove the existing uncertainty variable.
    (uncertainty_idx, uncertainty_units) = iom.remove_variable(var.uncertainty)

    # If the `--output-all-variables` option was specified, add the new baseline
    # and observed uncertainty variables.
    if args.output_all_variables:
      iom.insert_variable(
        uncertainty_idx, var.uncertainty + observed_suffix, uncertainty_units
      )
      iom.insert_variable(
        uncertainty_idx, var.uncertainty + baseline_suffix, uncertainty_units
      )

    try:
      # Remove the existing sample size variable.
      (sample_size_idx, sample_size_units) = iom.remove_variable(var.sample_size)

      # If the `--output-all-variables` option was specified, add the new
      # baseline and observed sample size variables.
      if args.output_all_variables:
        iom.insert_variable(
          sample_size_idx, var.sample_size + observed_suffix, sample_size_units
        )
        iom.insert_variable(
          sample_size_idx, var.sample_size + baseline_suffix, sample_size_units
        )
    except ValueError:
      # This is alright, because dependent variables may share the same sample
      # size variable.
      pass

  for var in args.control_variables:
    iom.remove_variable(var)

  # Add change variables.
  absolute_change_suffix = " - Change (`{0}` - `{1}`)".format(
    args.observed_input_file, args.baseline_input_file
  )

  percent_change_suffix = " - % Change (`{0}` to `{1}`)".format(
    args.observed_input_file, args.baseline_input_file
  )

  for var in dependent_variables:
    iom.append_variable(var.quantity + absolute_change_suffix, var.units)
    iom.append_variable(var.uncertainty + absolute_change_suffix, var.units)
    iom.append_variable(var.quantity + percent_change_suffix, "")
    iom.append_variable(var.uncertainty + percent_change_suffix, "")

  # Add all baseline input data to the `record_aggregator`.
  for record in iom.baseline():
    baseline_ra.append(record)
  
  for record in iom.observed():
    observed_ra.append(record)

  iom.write_header()

  # Compare and output results.
  for distinguishing_values, observed_dependent_values in observed_ra:
    try:
      baseline_dependent_values = baseline_ra[distinguishing_values]
    except KeyError: 
      assert False,                                                           \
        "Distinguishing value `"                                            + \
        str(baseline_ra.key_from_dict(distinguishing_values))               + \
        "` was not found in the baseline results."

    statistically_significant_change = False

    record = distinguishing_values.copy()

    # Compute changes, add the values and changes to the record, and identify
    # changes that are statistically significant.
    for var in dependent_variables:
      # Compute changes.
      baseline_quantity    = baseline_dependent_values[var.quantity]
      baseline_uncertainty = baseline_dependent_values[var.uncertainty]
      baseline_sample_size = baseline_dependent_values[var.sample_size]

      observed_quantity    = observed_dependent_values[var.quantity]
      observed_uncertainty = observed_dependent_values[var.uncertainty]
      observed_sample_size = observed_dependent_values[var.sample_size]

      (abs_change, abs_change_unc, per_change, per_change_unc) = \
        percent_change_uncertainty(
          baseline_quantity, baseline_uncertainty,
          observed_quantity, observed_uncertainty
        )

      # Round the change quantities and uncertainties to the significant digit
      # of uncertainty.
      try:
        abs_change_sigdig = max(
          find_significant_digit(abs_change),
          find_significant_digit(abs_change_unc),
        )

#        abs_change     = round_with_int_conversion(
#          abs_change,     abs_change_sigdig
#        )
#        abs_change_unc = round_with_int_conversion(
#          abs_change_unc, abs_change_sigdig
#        )
      except:
        # Any value errors should be due to NaNs returned by
        # `percent_change_uncertainty` because quantities or change in
        # quantities was 0. We can ignore these.
        pass

      try:
        per_change_sigdig = max(
          find_significant_digit(per_change),
          find_significant_digit(per_change_unc)
        )

#        per_change     = round_with_int_conversion(
#          per_change,     per_change_sigdig
#        )
#        per_change_unc = round_with_int_conversion(
#          per_change_unc, per_change_sigdig
#        )
      except:
        # Any value errors should be due to NaNs returned by
        # `percent_change_uncertainty` because quantities or change in
        # quantities was 0. We can ignore these.
        pass

      # Add the values (if the `--output-all-variables` option was specified)
      # and the changes to the record. Note that the record's schema is
      # different from the original schema. If multiple dependent variables
      # share the same sample size variable, it's fine - they will overwrite
      # each other, but with the same value.
      if args.output_all_variables:
        record[var.quantity + baseline_suffix]         = baseline_quantity
        record[var.uncertainty + baseline_suffix]      = baseline_uncertainty
        record[var.sample_size + baseline_suffix]      = baseline_sample_size
        record[var.quantity + observed_suffix]         = observed_quantity
        record[var.uncertainty + observed_suffix]      = observed_uncertainty
        record[var.sample_size + observed_suffix]      = observed_sample_size

      record[var.quantity + absolute_change_suffix]    = abs_change
      record[var.uncertainty + absolute_change_suffix] = abs_change_unc
      record[var.quantity + percent_change_suffix]     = per_change
      record[var.uncertainty + percent_change_suffix]  = per_change_unc

      # If the range of uncertainties overlap don't overlap and the percentage
      # change is greater than the change threshold, then change is
      # statistically significant.
      overlap = ranges_overlap_uncertainty(
          baseline_quantity, baseline_uncertainty,
          observed_quantity, observed_uncertainty
      )
      if not overlap and per_change >= args.change_threshold:
        statistically_significant_change = True

    # Print the record if a statistically significant change was found or if the
    # `--output-all-datapoints` option was specified.
    if args.output_all_datapoints or statistically_significant_change:
      iom.write(record)

