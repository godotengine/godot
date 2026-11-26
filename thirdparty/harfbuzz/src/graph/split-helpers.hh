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

#ifndef GRAPH_SPLIT_HELPERS_HH
#define GRAPH_SPLIT_HELPERS_HH

#include "graph-result.hh"

namespace graph {

template<typename Context>
HB_INTERNAL
graph_result_t<hb_vector_t<unsigned>> actuate_subtable_split (Context& split_context,
                                                              const hb_vector_t<unsigned>& split_points)
{
  hb_vector_t<unsigned> new_objects;
  if (!split_points)
    return Ok(new_objects);

  for (unsigned i = 0; i < split_points.length; i++)
  {
    unsigned start = split_points[i];
    unsigned end = (i < split_points.length - 1)
                   ? split_points[i + 1]
                   : split_context.original_count ();
    TRY_ASSIGN (unsigned id, split_context.clone_range (start, end));
    new_objects.push (id);
    TRY (graph_result_t<void>::from (new_objects, ALLOCATION_FAILURE));
  }

  TRY (split_context.shrink (split_points[0]));

  return Ok(new_objects);
}

}

#endif  // GRAPH_SPLIT_HELPERS_HH
