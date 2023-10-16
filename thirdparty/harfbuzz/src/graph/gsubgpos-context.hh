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

#include "graph.hh"
#include "../hb-ot-layout-gsubgpos.hh"

#ifndef GRAPH_GSUBGPOS_CONTEXT_HH
#define GRAPH_GSUBGPOS_CONTEXT_HH

namespace graph {

struct Lookup;

struct gsubgpos_graph_context_t
{
  hb_tag_t table_tag;
  graph_t& graph;
  unsigned lookup_list_index;
  hb_hashmap_t<unsigned, graph::Lookup*> lookups;


  HB_INTERNAL gsubgpos_graph_context_t (hb_tag_t table_tag_,
                                        graph_t& graph_);

  HB_INTERNAL unsigned create_node (unsigned size);

  bool add_buffer (char* buffer)
  {
    return graph.add_buffer (buffer);
  }

 private:
  HB_INTERNAL unsigned num_non_ext_subtables ();
};

}

#endif  // GRAPH_GSUBGPOS_CONTEXT
