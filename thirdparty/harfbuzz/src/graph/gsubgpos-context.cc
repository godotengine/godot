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

#include "gsubgpos-graph.hh"

namespace graph {

gsubgpos_graph_context_t::gsubgpos_graph_context_t (hb_tag_t table_tag_,
                                                    graph_t& graph_)
    : table_tag (table_tag_),
      graph (graph_),
      lookup_list_index (0),
      lookups ()
{
  if (table_tag_ != HB_OT_TAG_GPOS
      &&  table_tag_ != HB_OT_TAG_GSUB)
    return;

  GSTAR* gstar = graph::GSTAR::graph_to_gstar (graph_);
  if (gstar) {
    gstar->find_lookups (graph, lookups);
    lookup_list_index = gstar->get_lookup_list_index (graph_);
  }
}

unsigned gsubgpos_graph_context_t::create_node (unsigned size)
{
  char* buffer = (char*) hb_calloc (1, size);
  if (!buffer)
    return -1;

  if (!add_buffer (buffer)) {
    // Allocation did not get stored for freeing later.
    hb_free (buffer);
    return -1;
  }

  return graph.new_node (buffer, buffer + size);
}

unsigned gsubgpos_graph_context_t::num_non_ext_subtables ()  {
  unsigned count = 0;
  for (auto l : lookups.values ())
  {
    if (l->is_extension (table_tag)) continue;
    count += l->number_of_subtables ();
  }
  return count;
}

}
