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
#include "../OT/Layout/Common/Coverage.hh"

#ifndef GRAPH_COVERAGE_GRAPH_HH
#define GRAPH_COVERAGE_GRAPH_HH

namespace graph {

static bool sanitize (
  const OT::Layout::Common::CoverageFormat1_3<OT::Layout::SmallTypes>* thiz,
  graph_t::vertex_t& vertex
) {
  int64_t vertex_len = vertex.obj.tail - vertex.obj.head;
  constexpr unsigned min_size = OT::Layout::Common::CoverageFormat1_3<OT::Layout::SmallTypes>::min_size;
  if (vertex_len < min_size) return false;
  hb_barrier ();
  return vertex_len >= min_size + thiz->glyphArray.get_size () - thiz->glyphArray.len.get_size ();
}

static bool sanitize (
  const OT::Layout::Common::CoverageFormat2_4<OT::Layout::SmallTypes>* thiz,
  graph_t::vertex_t& vertex
) {
  int64_t vertex_len = vertex.obj.tail - vertex.obj.head;
  constexpr unsigned min_size = OT::Layout::Common::CoverageFormat2_4<OT::Layout::SmallTypes>::min_size;
  if (vertex_len < min_size) return false;
  hb_barrier ();
  return vertex_len >= min_size + thiz->rangeRecord.get_size () - thiz->rangeRecord.len.get_size ();
}

struct Coverage : public OT::Layout::Common::Coverage
{
  static Coverage* clone_coverage (gsubgpos_graph_context_t& c,
                                   unsigned coverage_id,
                                   unsigned new_parent_id,
                                   unsigned link_position,
                                   unsigned start, unsigned end)

  {
    unsigned coverage_size = c.graph.vertices_[coverage_id].table_size ();
    auto& coverage_v = c.graph.vertices_[coverage_id];
    Coverage* coverage_table = (Coverage*) coverage_v.obj.head;
    if (!coverage_table || !coverage_table->sanitize (coverage_v))
      return nullptr;

    auto new_coverage =
        + hb_zip (coverage_table->iter (), hb_range ())
        | hb_filter ([&] (hb_pair_t<unsigned, unsigned> p) {
          return p.second >= start && p.second < end;
        })
        | hb_map_retains_sorting (hb_first)
        ;

    return add_coverage (c, new_parent_id, link_position, new_coverage, coverage_size);
  }

  template<typename It>
  static Coverage* add_coverage (gsubgpos_graph_context_t& c,
                                 unsigned parent_id,
                                 unsigned link_position,
                                 It glyphs,
                                 unsigned max_size)
  {
    unsigned coverage_prime_id = c.graph.new_node (nullptr, nullptr);
    auto& coverage_prime_vertex = c.graph.vertices_[coverage_prime_id];
    if (!make_coverage (c, glyphs, coverage_prime_id, max_size))
      return nullptr;

    auto* coverage_link = c.graph.vertices_[parent_id].obj.real_links.push ();
    coverage_link->width = SmallTypes::size;
    coverage_link->objidx = coverage_prime_id;
    coverage_link->position = link_position;
    coverage_prime_vertex.add_parent (parent_id, false);

    return (Coverage*) coverage_prime_vertex.obj.head;
  }

  // Filter an existing coverage table to glyphs at indices [start, end) and replace it with the filtered version.
  static bool filter_coverage (gsubgpos_graph_context_t& c,
                               unsigned existing_coverage,
                               unsigned start, unsigned end) {
    unsigned coverage_size = c.graph.vertices_[existing_coverage].table_size ();
    auto& coverage_v = c.graph.vertices_[existing_coverage];
    Coverage* coverage_table = (Coverage*) coverage_v.obj.head;
    if (!coverage_table || !coverage_table->sanitize (coverage_v))
      return false;

    auto new_coverage =
        + hb_zip (coverage_table->iter (), hb_range ())
        | hb_filter ([&] (hb_pair_t<unsigned, unsigned> p) {
          return p.second >= start && p.second < end;
        })
        | hb_map_retains_sorting (hb_first)
        ;

    return make_coverage (c, new_coverage, existing_coverage, coverage_size * 2 + 100);
  }

  // Replace the coverage table at dest obj with one covering 'glyphs'.
  template<typename It>
  static bool make_coverage (gsubgpos_graph_context_t& c,
                             It glyphs,
                             unsigned dest_obj,
                             unsigned max_size)
  {
    char* buffer = (char*) hb_calloc (1, max_size);
    hb_serialize_context_t serializer (buffer, max_size);
    OT::Layout::Common::Coverage_serialize (&serializer, glyphs);
    serializer.end_serialize ();
    if (serializer.in_error ())
    {
      hb_free (buffer);
      return false;
    }

    hb_bytes_t coverage_copy = serializer.copy_bytes ();
    if (!coverage_copy.arrayZ) return false;
    // Give ownership to the context, it will cleanup the buffer.
    if (!c.add_buffer ((char *) coverage_copy.arrayZ))
    {
      hb_free ((char *) coverage_copy.arrayZ);
      return false;
    }

    auto& obj = c.graph.vertices_[dest_obj].obj;
    obj.head = (char *) coverage_copy.arrayZ;
    obj.tail = obj.head + coverage_copy.length;

    hb_free (buffer);
    return true;
  }

  bool sanitize (graph_t::vertex_t& vertex) const
  {
    int64_t vertex_len = vertex.obj.tail - vertex.obj.head;
    if (vertex_len < OT::Layout::Common::Coverage::min_size) return false;
    hb_barrier ();
    switch (u.format)
    {
    case 1: return graph::sanitize ((const OT::Layout::Common::CoverageFormat1_3<OT::Layout::SmallTypes>*) this, vertex);
    case 2: return graph::sanitize ((const OT::Layout::Common::CoverageFormat2_4<OT::Layout::SmallTypes>*) this, vertex);
#ifndef HB_NO_BEYOND_64K
    // Not currently supported
    case 3:
    case 4:
#endif
    default: return false;
    }
  }
};


}

#endif  // GRAPH_COVERAGE_GRAPH_HH
