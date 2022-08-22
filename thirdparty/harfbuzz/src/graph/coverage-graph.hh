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

struct CoverageFormat1 : public OT::Layout::Common::CoverageFormat1_3<SmallTypes>
{
  bool sanitize (graph_t::vertex_t& vertex) const
  {
    int64_t vertex_len = vertex.obj.tail - vertex.obj.head;
    constexpr unsigned min_size = OT::Layout::Common::CoverageFormat1_3<SmallTypes>::min_size;
    if (vertex_len < min_size) return false;
    return vertex_len >= min_size + glyphArray.get_size () - glyphArray.len.get_size ();
  }
};

struct CoverageFormat2 : public OT::Layout::Common::CoverageFormat2_4<SmallTypes>
{
  bool sanitize (graph_t::vertex_t& vertex) const
  {
    int64_t vertex_len = vertex.obj.tail - vertex.obj.head;
    constexpr unsigned min_size = OT::Layout::Common::CoverageFormat2_4<SmallTypes>::min_size;
    if (vertex_len < min_size) return false;
    return vertex_len >= min_size + rangeRecord.get_size () - rangeRecord.len.get_size ();
  }
};

struct Coverage : public OT::Layout::Common::Coverage
{
  bool sanitize (graph_t::vertex_t& vertex) const
  {
    int64_t vertex_len = vertex.obj.tail - vertex.obj.head;
    if (vertex_len < OT::Layout::Common::Coverage::min_size) return false;
    switch (u.format)
    {
    case 1: return ((CoverageFormat1*)this)->sanitize (vertex);
    case 2: return ((CoverageFormat2*)this)->sanitize (vertex);
#ifndef HB_NO_BORING_EXPANSION
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
