/*
 * Copyright © 2022  Google, Inc.
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 * Google Author(s): Garret Rieger
 */

#include "gsubgpos-context.hh"
#include "classdef-graph.hh"

typedef hb_pair_t<hb_codepoint_t, hb_codepoint_t> gid_and_class_t;
typedef hb_vector_t<gid_and_class_t> gid_and_class_list_t;


static bool incremental_size_is (const gid_and_class_list_t& list, unsigned klass,
                                 unsigned cov_expected, unsigned class_def_expected)
{
  graph::class_def_size_estimator_t estimator (list.iter ());

  unsigned result = estimator.incremental_coverage_size (klass);
  if (result != cov_expected)
  {
    printf ("FAIL: coverage expected size %u but was %u\n", cov_expected, result);
    return false;
  }

  result = estimator.incremental_class_def_size (klass);
  if (result != class_def_expected)
  {
    printf ("FAIL: class def expected size %u but was %u\n", class_def_expected, result);
    return false;
  }

  return true;
}

static void test_class_and_coverage_size_estimates ()
{
  gid_and_class_list_t empty = {
  };
  assert (incremental_size_is (empty, 0, 0, 0));
  assert (incremental_size_is (empty, 1, 0, 0));

  gid_and_class_list_t class_zero = {
    {5, 0},
  };
  assert (incremental_size_is (class_zero, 0, 2, 0));

  gid_and_class_list_t consecutive = {
    {4, 0},
    {5, 0},
    {6, 1},
    {7, 1},
    {8, 2},
    {9, 2},
    {10, 2},
    {11, 2},
  };
  assert (incremental_size_is (consecutive, 0, 4, 0));
  assert (incremental_size_is (consecutive, 1, 4, 4));
  assert (incremental_size_is (consecutive, 2, 8, 6));

  gid_and_class_list_t non_consecutive = {
    {4, 0},
    {5, 0},

    {6, 1},
    {7, 1},

    {9, 2},
    {10, 2},
    {11, 2},
    {12, 2},
  };
  assert (incremental_size_is (non_consecutive, 0, 4, 0));
  assert (incremental_size_is (non_consecutive, 1, 4, 6));
  assert (incremental_size_is (non_consecutive, 2, 8, 6));

  gid_and_class_list_t multiple_ranges = {
    {4, 0},
    {5, 0},

    {6, 1},
    {7, 1},

    {9, 1},

    {11, 1},
    {12, 1},
    {13, 1},
  };
  assert (incremental_size_is (multiple_ranges, 0, 4, 0));
  assert (incremental_size_is (multiple_ranges, 1, 2 * 6, 3 * 6));
}

int
main (int argc, char **argv)
{
  test_class_and_coverage_size_estimates ();
}
