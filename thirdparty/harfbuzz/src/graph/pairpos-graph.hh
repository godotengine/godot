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

#ifndef GRAPH_PAIRPOS_GRAPH_HH
#define GRAPH_PAIRPOS_GRAPH_HH

#include "coverage-graph.hh"
#include "../OT/Layout/GPOS/PairPos.hh"
#include "../OT/Layout/GPOS/PosLookupSubTable.hh"

namespace graph {

struct PairPosFormat1 : public OT::Layout::GPOS_impl::PairPosFormat1_3<SmallTypes>
{
  bool sanitize (graph_t::vertex_t& vertex) const
  {
    int64_t vertex_len = vertex.obj.tail - vertex.obj.head;
    unsigned min_size = OT::Layout::GPOS_impl::PairPosFormat1_3<SmallTypes>::min_size;
    if (vertex_len < min_size) return false;

    return vertex_len >=
        min_size + pairSet.get_size () - pairSet.len.get_size();
  }

  hb_vector_t<unsigned> split_subtables (gsubgpos_graph_context_t& c, unsigned this_index)
  {
    hb_set_t visited;

    const unsigned coverage_id = c.graph.index_for_offset (this_index, &coverage);
    const unsigned coverage_size = c.graph.vertices_[coverage_id].table_size ();
    const unsigned base_size = OT::Layout::GPOS_impl::PairPosFormat1_3<SmallTypes>::min_size
                               + coverage_size;

    unsigned accumulated = base_size;
    hb_vector_t<unsigned> split_points;
    for (unsigned i = 0; i < pairSet.len; i++)
    {
      unsigned pair_set_index = pair_set_graph_index (c, this_index, i);
      accumulated += c.graph.find_subgraph_size (pair_set_index, visited);
      accumulated += SmallTypes::size; // for PairSet offset.

      // TODO(garretrieger): don't count the size of the largest pairset against the limit, since
      //                     it will be packed last in the order and does not contribute to
      //                     the 64kb limit.

      if (accumulated > (1 << 16))
      {
        split_points.push (i);
        accumulated = base_size;
        visited.clear (); // Pretend node sharing isn't allowed between splits.
      }
    }

    return do_split (c, this_index, split_points);
  }

 private:

  // Split this PairPos into two or more PairPos's. split_points defines
  // the indices (first index to include in the new table) to split at.
  // Returns the object id's of the newly created PairPos subtables.
  hb_vector_t<unsigned> do_split (gsubgpos_graph_context_t& c,
                                  unsigned this_index,
                                  const hb_vector_t<unsigned> split_points)
  {
    hb_vector_t<unsigned> new_objects;
    if (!split_points)
      return new_objects;

    for (unsigned i = 0; i < split_points.length; i++)
    {
      unsigned start = split_points[i];
      unsigned end = (i < split_points.length - 1) ? split_points[i + 1] : pairSet.len;
      unsigned id = clone_range (c, this_index, start, end);

      if (id == (unsigned) -1)
      {
        new_objects.reset ();
        new_objects.allocated = -1; // mark error
        return new_objects;
      }
      new_objects.push (id);
    }

    if (!shrink (c, this_index, split_points[0]))
    {
      new_objects.reset ();
      new_objects.allocated = -1; // mark error
    }

    return new_objects;
  }

  bool shrink (gsubgpos_graph_context_t& c,
               unsigned this_index,
               unsigned count)
  {
    DEBUG_MSG (SUBSET_REPACK, nullptr,
               "  Shrinking PairPosFormat1 (%u) to [0, %u).",
               this_index,
               count);
    unsigned old_count = pairSet.len;
    if (count >= old_count)
      return true;

    pairSet.len = count;
    c.graph.vertices_[this_index].obj.tail -= (old_count - count) * SmallTypes::size;

    unsigned coverage_id = c.graph.index_for_offset (this_index, &coverage);
    unsigned coverage_size = c.graph.vertices_[coverage_id].table_size ();
    auto& coverage_v = c.graph.vertices_[coverage_id];
    Coverage* coverage_table = (Coverage*) coverage_v.obj.head;
    if (!coverage_table->sanitize (coverage_v))
      return false;

    auto new_coverage =
        + hb_zip (coverage_table->iter (), hb_range ())
        | hb_filter ([&] (hb_pair_t<unsigned, unsigned> p) {
          return p.second < count;
        })
        | hb_map_retains_sorting (hb_first)
        ;

    return make_coverage (c, new_coverage, coverage_id, coverage_size);
  }

  // Create a new PairPos including PairSet's from start (inclusive) to end (exclusive).
  // Returns object id of the new object.
  unsigned clone_range (gsubgpos_graph_context_t& c,
                        unsigned this_index,
                        unsigned start, unsigned end) const
  {
    DEBUG_MSG (SUBSET_REPACK, nullptr,
               "  Cloning PairPosFormat1 (%u) range [%u, %u).", this_index, start, end);

    unsigned num_pair_sets = end - start;
    unsigned prime_size = OT::Layout::GPOS_impl::PairPosFormat1_3<SmallTypes>::min_size
                          + num_pair_sets * SmallTypes::size;

    unsigned pair_pos_prime_id = c.create_node (prime_size);
    if (pair_pos_prime_id == (unsigned) -1) return -1;

    PairPosFormat1* pair_pos_prime = (PairPosFormat1*) c.graph.object (pair_pos_prime_id).head;
    pair_pos_prime->format = this->format;
    pair_pos_prime->valueFormat[0] = this->valueFormat[0];
    pair_pos_prime->valueFormat[1] = this->valueFormat[1];
    pair_pos_prime->pairSet.len = num_pair_sets;

    for (unsigned i = start; i < end; i++)
    {
      c.graph.move_child<> (this_index,
                            &pairSet[i],
                            pair_pos_prime_id,
                            &pair_pos_prime->pairSet[i - start]);
    }

    unsigned coverage_id = c.graph.index_for_offset (this_index, &coverage);
    unsigned coverage_size = c.graph.vertices_[coverage_id].table_size ();
    auto& coverage_v = c.graph.vertices_[coverage_id];
    Coverage* coverage_table = (Coverage*) coverage_v.obj.head;
    if (!coverage_table->sanitize (coverage_v))
      return false;

    auto new_coverage =
        + hb_zip (coverage_table->iter (), hb_range ())
        | hb_filter ([&] (hb_pair_t<unsigned, unsigned> p) {
          return p.second >= start && p.second < end;
        })
        | hb_map_retains_sorting (hb_first)
        ;

    unsigned coverage_prime_id = c.graph.new_node (nullptr, nullptr);
    auto& coverage_prime_vertex = c.graph.vertices_[coverage_prime_id];
    if (!make_coverage (c, new_coverage, coverage_prime_id, coverage_size))
      return -1;

    auto* coverage_link = c.graph.vertices_[pair_pos_prime_id].obj.real_links.push ();
    coverage_link->width = SmallTypes::size;
    coverage_link->objidx = coverage_prime_id;
    coverage_link->position = 2;
    coverage_prime_vertex.parents.push (pair_pos_prime_id);

    return pair_pos_prime_id;
  }

  template<typename It>
  bool make_coverage (gsubgpos_graph_context_t& c,
                      It glyphs,
                      unsigned dest_obj,
                      unsigned max_size) const
  {
    char* buffer = (char*) hb_calloc (1, max_size);
    hb_serialize_context_t serializer (buffer, max_size);
    Coverage_serialize (&serializer, glyphs);
    serializer.end_serialize ();
    if (serializer.in_error ())
    {
      hb_free (buffer);
      return false;
    }

    hb_bytes_t coverage_copy = serializer.copy_bytes ();
    c.add_buffer ((char *) coverage_copy.arrayZ); // Give ownership to the context, it will cleanup the buffer.

    auto& obj = c.graph.vertices_[dest_obj].obj;
    obj.head = (char *) coverage_copy.arrayZ;
    obj.tail = obj.head + coverage_copy.length;

    hb_free (buffer);
    return true;
  }

  unsigned pair_set_graph_index (gsubgpos_graph_context_t& c, unsigned this_index, unsigned i) const
  {
    return c.graph.index_for_offset (this_index, &pairSet[i]);
  }
};

struct PairPosFormat2 : public OT::Layout::GPOS_impl::PairPosFormat2_4<SmallTypes>
{
  bool sanitize (graph_t::vertex_t& vertex) const
  {
    // TODO(garretrieger): implement me!
    return true;
  }

  hb_vector_t<unsigned> split_subtables (gsubgpos_graph_context_t& c, unsigned this_index)
  {
    // TODO(garretrieger): implement me!
    return hb_vector_t<unsigned> ();
  }
};

struct PairPos : public OT::Layout::GPOS_impl::PairPos
{
  hb_vector_t<unsigned> split_subtables (gsubgpos_graph_context_t& c, unsigned this_index)
  {
    switch (u.format) {
    case 1:
      return ((PairPosFormat1*)(&u.format1))->split_subtables (c, this_index);
    case 2:
      return ((PairPosFormat2*)(&u.format2))->split_subtables (c, this_index);
#ifndef HB_NO_BORING_EXPANSION
    case 3: HB_FALLTHROUGH;
    case 4: HB_FALLTHROUGH;
      // Don't split 24bit PairPos's.
#endif
    default:
      return hb_vector_t<unsigned> ();
    }
  }

  bool sanitize (graph_t::vertex_t& vertex) const
  {
    int64_t vertex_len = vertex.obj.tail - vertex.obj.head;
    if (vertex_len < u.format.get_size ()) return false;

    switch (u.format) {
    case 1:
      return ((PairPosFormat1*)(&u.format1))->sanitize (vertex);
    case 2:
      return ((PairPosFormat2*)(&u.format2))->sanitize (vertex);
#ifndef HB_NO_BORING_EXPANSION
    case 3: HB_FALLTHROUGH;
    case 4: HB_FALLTHROUGH;
#endif
    default:
      // We don't handle format 3 and 4 here.
      return false;
    }
  }
};

}

#endif  // GRAPH_PAIRPOS_GRAPH_HH
