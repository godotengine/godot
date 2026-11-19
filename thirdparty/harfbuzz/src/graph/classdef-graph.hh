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
#include "../hb-ot-layout-common.hh"
#include "graph-result.hh"

#ifndef GRAPH_CLASSDEF_GRAPH_HH
#define GRAPH_CLASSDEF_GRAPH_HH

namespace graph {

template <typename Types>
struct ClassDefFormat1_3 : public OT::ClassDefFormat1_3<Types>
{
  graph_result_t<void> sanitize (const graph_t::vertex_t& vertex) const
  {
    size_t vertex_len = vertex.table_size();
    constexpr unsigned min_size = OT::ClassDefFormat1_3<Types>::min_size;
    if (unlikely (vertex_len < min_size)) return Err(SANITIZE_FAILURE);
    hb_barrier ();
    if (unlikely (vertex_len < min_size + this->classValue.get_size () - this->classValue.len.get_size ()))
      return Err(SANITIZE_FAILURE);
    return Ok();
  }
};

template <typename Types>
struct ClassDefFormat2_4 : public OT::ClassDefFormat2_4<Types>
{
  graph_result_t<void> sanitize (const graph_t::vertex_t& vertex) const
  {
    size_t vertex_len = vertex.table_size();
    constexpr unsigned min_size = OT::ClassDefFormat2_4<Types>::min_size;
    if (unlikely (vertex_len < min_size)) return Err(SANITIZE_FAILURE);
    hb_barrier ();
    if (unlikely (vertex_len < min_size + this->rangeRecord.get_size () - this->rangeRecord.len.get_size ()))
      return Err(SANITIZE_FAILURE);
    return Ok();
  }
};

struct ClassDef : public OT::ClassDef
{
  template<typename It>
  static graph_result_t<void> add_class_def (gsubgpos_graph_context_t& c,
                                             unsigned parent_id,
                                             unsigned link_position,
                                             It glyph_and_class,
                                             unsigned max_size)
  {
    TRY_ASSIGN (unsigned class_def_prime_id, c.graph.new_node (nullptr, nullptr));
    auto& class_def_prime_vertex = c.graph.vertices_[class_def_prime_id];
    TRY (make_class_def (c, glyph_and_class, class_def_prime_id, max_size));

    TRY(c.graph.vertices_[parent_id].add_real_link (SmallTypes::size, class_def_prime_id, link_position));
    TRY(class_def_prime_vertex.add_parent (parent_id, false));

    return Ok();
  }

  template<typename It>
  static graph_result_t<void> make_class_def (gsubgpos_graph_context_t& c,
                                              It glyph_and_class,
                                              unsigned dest_obj,
                                              unsigned max_size)
  {
    char* buffer = (char*) hb_calloc (1, max_size);
    if (unlikely (!buffer))
      return Err(ALLOCATION_FAILURE);

    hb_serialize_context_t serializer (buffer, max_size);
    OT::ClassDef_serialize (&serializer, glyph_and_class);
    serializer.end_serialize ();
    if (unlikely (serializer.in_error ()))
    {
      hb_free (buffer);
      return Err(ALLOCATION_FAILURE);
    }

    hb_bytes_t class_def_copy = serializer.copy_bytes ();
    if (unlikely (!class_def_copy.arrayZ))
    {
      hb_free (buffer);
      return Err(ALLOCATION_FAILURE);
    }

    // Give ownership to the context, it will cleanup the buffer.
    auto res = c.add_buffer ((char *) class_def_copy.arrayZ);
    if (unlikely (!res.is_ok ()))
    {
      hb_free (buffer);
      hb_free ((char *) class_def_copy.arrayZ);
      return res;
    }


    auto& v = c.graph.vertices_[dest_obj];
    char* head = (char *) class_def_copy.arrayZ;
    v.set_buffer (head, head + class_def_copy.length);

    hb_free (buffer);
    return Ok();
  }

  graph_result_t<void> sanitize (const graph_t::vertex_t& vertex) const
  {
    size_t vertex_len = vertex.table_size();
    if (unlikely (vertex_len < OT::ClassDef::min_size)) return Err(SANITIZE_FAILURE);
    hb_barrier ();
    switch (u.format.v)
    {
    case 1: return ((ClassDefFormat1_3<SmallTypes>*)this)->sanitize (vertex);
    case 2: return ((ClassDefFormat2_4<SmallTypes>*)this)->sanitize (vertex);
#ifndef HB_NO_BEYOND_64K
    case 3: return ((ClassDefFormat1_3<MediumTypes>*)this)->sanitize (vertex);
    case 4: return ((ClassDefFormat2_4<MediumTypes>*)this)->sanitize (vertex);
#endif
    default: return Err(SANITIZE_FAILURE);
    }
  }
};


struct class_def_size_estimator_t
{
  template<typename It>
  static graph_result_t<class_def_size_estimator_t> create (It glyph_and_class)
  {
    class_def_size_estimator_t estimator;
    estimator.reset();
    for (auto p : + glyph_and_class)
    {
      unsigned gid = p.first;
      unsigned klass = p.second;

      hb_set_t* glyphs;
      if (estimator.glyphs_per_class.has (klass, &glyphs) && glyphs) {
        glyphs->add (gid);
        continue;
      }

      hb_set_t new_glyphs;
      new_glyphs.add (gid);
      estimator.glyphs_per_class.set (klass, std::move (new_glyphs));
    }

    TRY(estimator.to_result ());

    for (unsigned klass : estimator.glyphs_per_class.keys ())
    {
      if (!klass) continue; // class 0 doesn't get encoded.

      const hb_set_t& glyphs = estimator.glyphs_per_class.get (klass);
      hb_codepoint_t start = HB_SET_VALUE_INVALID;
      hb_codepoint_t end = HB_SET_VALUE_INVALID;

      unsigned count = 0;
      while (glyphs.next_range (&start, &end))
        count++;

      estimator.num_ranges_per_class.set (klass, count);
    }

    TRY(estimator.to_result ());
    return estimator;
  }

  void reset() {
    included_glyphs.clear();
    included_class_def_glyphs.clear();
    included_classes.clear();
    class_def_range_count = 0;
    class_max = 0;
    coverage_size_val = compute_coverage_size ();
  }

  // Size of coverage for all glyphs added via 'add_class_def_size'.
  unsigned coverage_size () const
  {
    return coverage_size_val;
  }

  // Compute the new size of the ClassDef table if all glyphs associated with 'klass' were added.
  unsigned add_class_def_size (unsigned klass)
  {
    if (!included_classes.has(klass)) {
      hb_set_t* glyphs = nullptr;
      if (glyphs_per_class.has(klass, &glyphs)) {
        unsigned num_glyphs = included_glyphs.get_population();
        included_glyphs.union_(*glyphs);
        if (klass)
          included_class_def_glyphs.union_(*glyphs);
        if (num_glyphs != included_glyphs.get_population())
          coverage_size_val = compute_coverage_size ();
      }

      if (klass)
      {
        class_def_range_count += num_ranges_per_class.get (klass);
        if (glyphs && klass > class_max)
          class_max = klass;
      }
      included_classes.add(klass);
    }

    return class_def_size ();
  }

  unsigned num_glyph_ranges (const hb_set_t &glyphs) const {
    hb_codepoint_t start = HB_SET_VALUE_INVALID;
    hb_codepoint_t end = HB_SET_VALUE_INVALID;

    unsigned count = 0;
    while (glyphs.next_range (&start, &end)) {
        count++;
    }
    return count;
  }

 private:
  graph_result_t<void> to_result() const
  {
    TRY (graph_result_t<void>::from (num_ranges_per_class, ALLOCATION_FAILURE));
    TRY (graph_result_t<void>::from (glyphs_per_class, ALLOCATION_FAILURE));
    TRY (graph_result_t<void>::from (included_classes, ALLOCATION_FAILURE));
    TRY (graph_result_t<void>::from (included_glyphs, ALLOCATION_FAILURE));
    TRY (graph_result_t<void>::from (included_class_def_glyphs, ALLOCATION_FAILURE));

    for (const hb_set_t& s : glyphs_per_class.values ())
    {
          TRY (graph_result_t<void>::from (s, ALLOCATION_FAILURE));
    }
    return Ok();
  }

  class_def_size_estimator_t ()
      : num_ranges_per_class (), glyphs_per_class () {}

  unsigned compute_coverage_size () const
  {
    unsigned glyph_size = glyph_id_size (included_glyphs);
    unsigned coverage_base_size = OT::HBUINT16::static_size + glyph_size;
    unsigned range_size = 3 * glyph_size;
    unsigned format1_size = coverage_base_size + glyph_size * included_glyphs.get_population();
    unsigned format2_size = coverage_base_size + range_size * num_glyph_ranges (included_glyphs);
    return hb_min(format1_size, format2_size);
  }

  unsigned glyph_id_size (const hb_set_t &glyphs) const
  {
#ifndef HB_NO_BEYOND_64K
    if (!glyphs.is_empty () && glyphs.get_max () > 0xFFFFu)
      return OT::Layout::MediumTypes::size;
#endif
    return OT::Layout::SmallTypes::size;
  }

  unsigned class_def_size () const
  {
    if (included_class_def_glyphs.is_empty ())
      return OT::HBUINT16::static_size + OT::Layout::SmallTypes::size;

    unsigned best_size = UINT_MAX;
    hb_codepoint_t glyph_max = included_class_def_glyphs.get_max ();
    unsigned glyph_span = glyph_max - included_class_def_glyphs.get_min () + 1;

    if (glyph_max <= 0xFFFFu && glyph_span <= 0xFFFFu && class_max <= 0xFFFFu)
      best_size = hb_min (best_size, 6 + 2 * glyph_span);
    if (glyph_max <= 0xFFFFu && class_def_range_count <= 0xFFFFu && class_max <= 0xFFFFu)
      best_size = hb_min (best_size, 4 + 6 * class_def_range_count);
#ifndef HB_NO_BEYOND_64K
    if (glyph_max <= 0xFFFFFFu && glyph_span <= 0xFFFFFFu && class_max <= 0xFFFFFFu)
      best_size = hb_min (best_size, 8 + 3 * glyph_span);
    if (glyph_max <= 0xFFFFFFu && class_def_range_count <= 0xFFFFFFu && class_max <= 0xFFFFu)
      best_size = hb_min (best_size, 5 + 8 * class_def_range_count);
#endif
    return best_size;
  }

  hb_hashmap_t<unsigned, unsigned> num_ranges_per_class;
  hb_hashmap_t<unsigned, hb_set_t> glyphs_per_class;
  hb_set_t included_classes;
  hb_set_t included_glyphs;
  hb_set_t included_class_def_glyphs;
  unsigned class_def_range_count;
  unsigned class_max;
  unsigned coverage_size_val;
};


}

#endif  // GRAPH_CLASSDEF_GRAPH_HH
