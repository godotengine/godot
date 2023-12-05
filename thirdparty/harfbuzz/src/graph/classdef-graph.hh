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

#ifndef GRAPH_CLASSDEF_GRAPH_HH
#define GRAPH_CLASSDEF_GRAPH_HH

namespace graph {

struct ClassDefFormat1 : public OT::ClassDefFormat1_3<SmallTypes>
{
  bool sanitize (graph_t::vertex_t& vertex) const
  {
    int64_t vertex_len = vertex.obj.tail - vertex.obj.head;
    constexpr unsigned min_size = OT::ClassDefFormat1_3<SmallTypes>::min_size;
    if (vertex_len < min_size) return false;
    return vertex_len >= min_size + classValue.get_size () - classValue.len.get_size ();
  }
};

struct ClassDefFormat2 : public OT::ClassDefFormat2_4<SmallTypes>
{
  bool sanitize (graph_t::vertex_t& vertex) const
  {
    int64_t vertex_len = vertex.obj.tail - vertex.obj.head;
    constexpr unsigned min_size = OT::ClassDefFormat2_4<SmallTypes>::min_size;
    if (vertex_len < min_size) return false;
    return vertex_len >= min_size + rangeRecord.get_size () - rangeRecord.len.get_size ();
  }
};

struct ClassDef : public OT::ClassDef
{
  template<typename It>
  static bool add_class_def (gsubgpos_graph_context_t& c,
                             unsigned parent_id,
                             unsigned link_position,
                             It glyph_and_class,
                             unsigned max_size)
  {
    unsigned class_def_prime_id = c.graph.new_node (nullptr, nullptr);
    auto& class_def_prime_vertex = c.graph.vertices_[class_def_prime_id];
    if (!make_class_def (c, glyph_and_class, class_def_prime_id, max_size))
      return false;

    auto* class_def_link = c.graph.vertices_[parent_id].obj.real_links.push ();
    class_def_link->width = SmallTypes::size;
    class_def_link->objidx = class_def_prime_id;
    class_def_link->position = link_position;
    class_def_prime_vertex.add_parent (parent_id);

    return true;
  }

  template<typename It>
  static bool make_class_def (gsubgpos_graph_context_t& c,
                              It glyph_and_class,
                              unsigned dest_obj,
                              unsigned max_size)
  {
    char* buffer = (char*) hb_calloc (1, max_size);
    hb_serialize_context_t serializer (buffer, max_size);
    OT::ClassDef_serialize (&serializer, glyph_and_class);
    serializer.end_serialize ();
    if (serializer.in_error ())
    {
      hb_free (buffer);
      return false;
    }

    hb_bytes_t class_def_copy = serializer.copy_bytes ();
    if (!class_def_copy.arrayZ) return false;
    // Give ownership to the context, it will cleanup the buffer.
    if (!c.add_buffer ((char *) class_def_copy.arrayZ))
    {
      hb_free ((char *) class_def_copy.arrayZ);
      return false;
    }

    auto& obj = c.graph.vertices_[dest_obj].obj;
    obj.head = (char *) class_def_copy.arrayZ;
    obj.tail = obj.head + class_def_copy.length;

    hb_free (buffer);
    return true;
  }

  bool sanitize (graph_t::vertex_t& vertex) const
  {
    int64_t vertex_len = vertex.obj.tail - vertex.obj.head;
    if (vertex_len < OT::ClassDef::min_size) return false;
    switch (u.format)
    {
    case 1: return ((ClassDefFormat1*)this)->sanitize (vertex);
    case 2: return ((ClassDefFormat2*)this)->sanitize (vertex);
#ifndef HB_NO_BEYOND_64K
    // Not currently supported
    case 3:
    case 4:
#endif
    default: return false;
    }
  }
};


struct class_def_size_estimator_t
{
  template<typename It>
  class_def_size_estimator_t (It glyph_and_class)
      : gids_consecutive (true), num_ranges_per_class (), glyphs_per_class ()
  {
    unsigned last_gid = (unsigned) -1;
    for (auto p : + glyph_and_class)
    {
      unsigned gid = p.first;
      unsigned klass = p.second;

      if (last_gid != (unsigned) -1 && gid != last_gid + 1)
        gids_consecutive = false;
      last_gid = gid;

      hb_set_t* glyphs;
      if (glyphs_per_class.has (klass, &glyphs) && glyphs) {
        glyphs->add (gid);
        continue;
      }

      hb_set_t new_glyphs;
      new_glyphs.add (gid);
      glyphs_per_class.set (klass, std::move (new_glyphs));
    }

    if (in_error ()) return;

    for (unsigned klass : glyphs_per_class.keys ())
    {
      if (!klass) continue; // class 0 doesn't get encoded.

      const hb_set_t& glyphs = glyphs_per_class.get (klass);
      hb_codepoint_t start = HB_SET_VALUE_INVALID;
      hb_codepoint_t end = HB_SET_VALUE_INVALID;

      unsigned count = 0;
      while (glyphs.next_range (&start, &end))
        count++;

      num_ranges_per_class.set (klass, count);
    }
  }

  // Incremental increase in the Coverage and ClassDef table size
  // (worst case) if all glyphs associated with 'klass' were added.
  unsigned incremental_coverage_size (unsigned klass) const
  {
    // Coverage takes 2 bytes per glyph worst case,
    return 2 * glyphs_per_class.get (klass).get_population ();
  }

  // Incremental increase in the Coverage and ClassDef table size
  // (worst case) if all glyphs associated with 'klass' were added.
  unsigned incremental_class_def_size (unsigned klass) const
  {
    // ClassDef takes 6 bytes per range
    unsigned class_def_2_size = 6 * num_ranges_per_class.get (klass);
    if (gids_consecutive)
    {
      // ClassDef1 takes 2 bytes per glyph, but only can be used
      // when gids are consecutive.
      return hb_min (2 * glyphs_per_class.get (klass).get_population (), class_def_2_size);
    }

    return class_def_2_size;
  }

  bool in_error ()
  {
    if (num_ranges_per_class.in_error ()) return true;
    if (glyphs_per_class.in_error ()) return true;

    for (const hb_set_t& s : glyphs_per_class.values ())
    {
      if (s.in_error ()) return true;
    }
    return false;
  }

 private:
  bool gids_consecutive;
  hb_hashmap_t<unsigned, unsigned> num_ranges_per_class;
  hb_hashmap_t<unsigned, hb_set_t> glyphs_per_class;
};


}

#endif  // GRAPH_CLASSDEF_GRAPH_HH
