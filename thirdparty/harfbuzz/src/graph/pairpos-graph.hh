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

#include "split-helpers.hh"
#include "coverage-graph.hh"
#include "classdef-graph.hh"
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
    hb_barrier ();

    return vertex_len >=
        min_size + pairSet.get_size () - pairSet.len.get_size();
  }

  hb_vector_t<unsigned> split_subtables (gsubgpos_graph_context_t& c,
                                         unsigned parent_index,
                                         unsigned this_index)
  {
    hb_set_t visited;

    const unsigned coverage_id = c.graph.index_for_offset (this_index, &coverage);
    const unsigned coverage_size = c.graph.vertices_[coverage_id].table_size ();
    const unsigned base_size = OT::Layout::GPOS_impl::PairPosFormat1_3<SmallTypes>::min_size;

    unsigned partial_coverage_size = 4;
    unsigned accumulated = base_size;
    hb_vector_t<unsigned> split_points;
    for (unsigned i = 0; i < pairSet.len; i++)
    {
      unsigned pair_set_index = pair_set_graph_index (c, this_index, i);
      unsigned accumulated_delta =
          c.graph.find_subgraph_size (pair_set_index, visited) +
          SmallTypes::size; // for PairSet offset.
      partial_coverage_size += OT::HBUINT16::static_size;

      accumulated += accumulated_delta;
      unsigned total = accumulated + hb_min (partial_coverage_size, coverage_size);

      if (total >= (1 << 16))
      {
        split_points.push (i);
        accumulated = base_size + accumulated_delta;
        partial_coverage_size = 6;
        visited.clear (); // node sharing isn't allowed between splits.
      }
    }

    split_context_t split_context {
      c,
      this,
      c.graph.duplicate_if_shared (parent_index, this_index),
    };

    return actuate_subtable_split<split_context_t> (split_context, split_points);
  }

 private:

  struct split_context_t {
    gsubgpos_graph_context_t& c;
    PairPosFormat1* thiz;
    unsigned this_index;

    unsigned original_count ()
    {
      return thiz->pairSet.len;
    }

    unsigned clone_range (unsigned start, unsigned end)
    {
      return thiz->clone_range (this->c, this->this_index, start, end);
    }

    bool shrink (unsigned count)
    {
      return thiz->shrink (this->c, this->this_index, count);
    }
  };

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

    auto coverage = c.graph.as_mutable_table<Coverage> (this_index, &this->coverage);
    if (!coverage) return false;

    unsigned coverage_size = coverage.vertex->table_size ();
    auto new_coverage =
        + hb_zip (coverage.table->iter (), hb_range ())
        | hb_filter ([&] (hb_pair_t<unsigned, unsigned> p) {
          return p.second < count;
        })
        | hb_map_retains_sorting (hb_first)
        ;

    return Coverage::make_coverage (c, new_coverage, coverage.index, coverage_size);
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
    if (!Coverage::clone_coverage (c,
                                   coverage_id,
                                   pair_pos_prime_id,
                                   2,
                                   start, end))
      return -1;

    return pair_pos_prime_id;
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
    size_t vertex_len = vertex.table_size ();
    unsigned min_size = OT::Layout::GPOS_impl::PairPosFormat2_4<SmallTypes>::min_size;
    if (vertex_len < min_size) return false;
    hb_barrier ();

    const unsigned class1_count = class1Count;
    return vertex_len >=
        min_size + class1_count * get_class1_record_size ();
  }

  hb_vector_t<unsigned> split_subtables (gsubgpos_graph_context_t& c,
                                         unsigned parent_index,
                                         unsigned this_index)
  {
    const unsigned base_size = OT::Layout::GPOS_impl::PairPosFormat2_4<SmallTypes>::min_size;
    const unsigned class_def_2_size = size_of (c, this_index, &classDef2);
    const Coverage* coverage = get_coverage (c, this_index);
    const ClassDef* class_def_1 = get_class_def_1 (c, this_index);
    auto gid_and_class =
        + coverage->iter ()
        | hb_map_retains_sorting ([&] (hb_codepoint_t gid) {
          return hb_codepoint_pair_t (gid, class_def_1->get_class (gid));
        })
        ;
    class_def_size_estimator_t estimator (gid_and_class);

    const unsigned class1_count = class1Count;
    const unsigned class2_count = class2Count;
    const unsigned class1_record_size = get_class1_record_size ();

    const unsigned value_1_len = valueFormat1.get_len ();
    const unsigned value_2_len = valueFormat2.get_len ();
    const unsigned total_value_len = value_1_len + value_2_len;

    unsigned accumulated = base_size;
    unsigned coverage_size = 4;
    unsigned class_def_1_size = 4;
    unsigned max_coverage_size = coverage_size;
    unsigned max_class_def_1_size = class_def_1_size;

    hb_vector_t<unsigned> split_points;

    hb_hashmap_t<unsigned, unsigned> device_tables = get_all_device_tables (c, this_index);
    hb_vector_t<unsigned> format1_device_table_indices = valueFormat1.get_device_table_indices ();
    hb_vector_t<unsigned> format2_device_table_indices = valueFormat2.get_device_table_indices ();
    bool has_device_tables = bool(format1_device_table_indices) || bool(format2_device_table_indices);

    hb_set_t visited;
    for (unsigned i = 0; i < class1_count; i++)
    {
      unsigned accumulated_delta = class1_record_size;
      coverage_size += estimator.incremental_coverage_size (i);
      class_def_1_size += estimator.incremental_class_def_size (i);
      max_coverage_size = hb_max (max_coverage_size, coverage_size);
      max_class_def_1_size = hb_max (max_class_def_1_size, class_def_1_size);

      if (has_device_tables) {
        for (unsigned j = 0; j < class2_count; j++)
        {
          unsigned value1_index = total_value_len * (class2_count * i + j);
          unsigned value2_index = value1_index + value_1_len;
          accumulated_delta += size_of_value_record_children (c,
                                                        device_tables,
                                                        format1_device_table_indices,
                                                        value1_index,
                                                        visited);
          accumulated_delta += size_of_value_record_children (c,
                                                        device_tables,
                                                        format2_device_table_indices,
                                                        value2_index,
                                                        visited);
        }
      }

      accumulated += accumulated_delta;
      unsigned total = accumulated
                       + coverage_size + class_def_1_size + class_def_2_size
                       // The largest object will pack last and can exceed the size limit.
                       - hb_max (hb_max (coverage_size, class_def_1_size), class_def_2_size);
      if (total >= (1 << 16))
      {
        split_points.push (i);
        // split does not include i, so add the size for i when we reset the size counters.
        accumulated = base_size + accumulated_delta;
        coverage_size = 4 + estimator.incremental_coverage_size (i);
        class_def_1_size = 4 + estimator.incremental_class_def_size (i);
        visited.clear (); // node sharing isn't allowed between splits.
      }
    }

    split_context_t split_context {
      c,
      this,
      c.graph.duplicate_if_shared (parent_index, this_index),
      class1_record_size,
      total_value_len,
      value_1_len,
      value_2_len,
      max_coverage_size,
      max_class_def_1_size,
      device_tables,
      format1_device_table_indices,
      format2_device_table_indices
    };

    return actuate_subtable_split<split_context_t> (split_context, split_points);
  }
 private:

  struct split_context_t
  {
    gsubgpos_graph_context_t& c;
    PairPosFormat2* thiz;
    unsigned this_index;
    unsigned class1_record_size;
    unsigned value_record_len;
    unsigned value1_record_len;
    unsigned value2_record_len;
    unsigned max_coverage_size;
    unsigned max_class_def_size;

    const hb_hashmap_t<unsigned, unsigned>& device_tables;
    const hb_vector_t<unsigned>& format1_device_table_indices;
    const hb_vector_t<unsigned>& format2_device_table_indices;

    unsigned original_count ()
    {
      return thiz->class1Count;
    }

    unsigned clone_range (unsigned start, unsigned end)
    {
      return thiz->clone_range (*this, start, end);
    }

    bool shrink (unsigned count)
    {
      return thiz->shrink (*this, count);
    }
  };

  size_t get_class1_record_size () const
  {
    const size_t class2_count = class2Count;
    return
        class2_count * (valueFormat1.get_size () + valueFormat2.get_size ());
  }

  unsigned clone_range (split_context_t& split_context,
                        unsigned start, unsigned end) const
  {
    DEBUG_MSG (SUBSET_REPACK, nullptr,
               "  Cloning PairPosFormat2 (%u) range [%u, %u).", split_context.this_index, start, end);

    graph_t& graph = split_context.c.graph;

    unsigned num_records = end - start;
    unsigned prime_size = OT::Layout::GPOS_impl::PairPosFormat2_4<SmallTypes>::min_size
                          + num_records * split_context.class1_record_size;

    unsigned pair_pos_prime_id = split_context.c.create_node (prime_size);
    if (pair_pos_prime_id == (unsigned) -1) return -1;

    PairPosFormat2* pair_pos_prime =
        (PairPosFormat2*) graph.object (pair_pos_prime_id).head;
    pair_pos_prime->format = this->format;
    pair_pos_prime->valueFormat1 = this->valueFormat1;
    pair_pos_prime->valueFormat2 = this->valueFormat2;
    pair_pos_prime->class1Count = num_records;
    pair_pos_prime->class2Count = this->class2Count;
    clone_class1_records (split_context,
                          pair_pos_prime_id,
                          start,
                          end);

    unsigned coverage_id =
        graph.index_for_offset (split_context.this_index, &coverage);
    unsigned class_def_1_id =
        graph.index_for_offset (split_context.this_index, &classDef1);
    auto& coverage_v = graph.vertices_[coverage_id];
    auto& class_def_1_v = graph.vertices_[class_def_1_id];
    Coverage* coverage_table = (Coverage*) coverage_v.obj.head;
    ClassDef* class_def_1_table = (ClassDef*) class_def_1_v.obj.head;
    if (!coverage_table
        || !coverage_table->sanitize (coverage_v)
        || !class_def_1_table
        || !class_def_1_table->sanitize (class_def_1_v))
      return -1;

    auto klass_map =
    + coverage_table->iter ()
    | hb_map_retains_sorting ([&] (hb_codepoint_t gid) {
      return hb_codepoint_pair_t (gid, class_def_1_table->get_class (gid));
    })
    | hb_filter ([&] (hb_codepoint_t klass) {
      return klass >= start && klass < end;
    }, hb_second)
    | hb_map_retains_sorting ([&] (hb_codepoint_pair_t gid_and_class) {
      // Classes must be from 0...N so subtract start
      return hb_codepoint_pair_t (gid_and_class.first, gid_and_class.second - start);
    })
    ;

    if (!Coverage::add_coverage (split_context.c,
                                 pair_pos_prime_id,
                                 2,
                                 + klass_map | hb_map_retains_sorting (hb_first),
                                 split_context.max_coverage_size))
      return -1;

    // classDef1
    if (!ClassDef::add_class_def (split_context.c,
                                  pair_pos_prime_id,
                                  8,
                                  + klass_map,
                                  split_context.max_class_def_size))
      return -1;

    // classDef2
    unsigned class_def_2_id =
        graph.index_for_offset (split_context.this_index, &classDef2);
    auto* class_def_link = graph.vertices_[pair_pos_prime_id].obj.real_links.push ();
    class_def_link->width = SmallTypes::size;
    class_def_link->objidx = class_def_2_id;
    class_def_link->position = 10;
    graph.vertices_[class_def_2_id].add_parent (pair_pos_prime_id);
    graph.duplicate (pair_pos_prime_id, class_def_2_id);

    return pair_pos_prime_id;
  }

  void clone_class1_records (split_context_t& split_context,
                             unsigned pair_pos_prime_id,
                             unsigned start, unsigned end) const
  {
    PairPosFormat2* pair_pos_prime =
        (PairPosFormat2*) split_context.c.graph.object (pair_pos_prime_id).head;

    char* start_addr = ((char*)&values[0]) + start * split_context.class1_record_size;
    unsigned num_records = end - start;
    hb_memcpy (&pair_pos_prime->values[0],
            start_addr,
            num_records * split_context.class1_record_size);

    if (!split_context.format1_device_table_indices
        && !split_context.format2_device_table_indices)
      // No device tables to move over.
      return;

    unsigned class2_count = class2Count;
    for (unsigned i = start; i < end; i++)
    {
      for (unsigned j = 0; j < class2_count; j++)
      {
        unsigned value1_index = split_context.value_record_len * (class2_count * i + j);
        unsigned value2_index = value1_index + split_context.value1_record_len;

        unsigned new_value1_index = split_context.value_record_len * (class2_count * (i - start) + j);
        unsigned new_value2_index = new_value1_index + split_context.value1_record_len;

        transfer_device_tables (split_context,
                                pair_pos_prime_id,
                                split_context.format1_device_table_indices,
                                value1_index,
                                new_value1_index);

        transfer_device_tables (split_context,
                                pair_pos_prime_id,
                                split_context.format2_device_table_indices,
                                value2_index,
                                new_value2_index);
      }
    }
  }

  void transfer_device_tables (split_context_t& split_context,
                               unsigned pair_pos_prime_id,
                               const hb_vector_t<unsigned>& device_table_indices,
                               unsigned old_value_record_index,
                               unsigned new_value_record_index) const
  {
    PairPosFormat2* pair_pos_prime =
        (PairPosFormat2*) split_context.c.graph.object (pair_pos_prime_id).head;

    for (unsigned i : device_table_indices)
    {
      OT::Offset16* record = (OT::Offset16*) &values[old_value_record_index + i];
      unsigned record_position = ((char*) record) - ((char*) this);
      if (!split_context.device_tables.has (record_position)) continue;

      split_context.c.graph.move_child (
          split_context.this_index,
          record,
          pair_pos_prime_id,
          (OT::Offset16*) &pair_pos_prime->values[new_value_record_index + i]);
    }
  }

  bool shrink (split_context_t& split_context,
               unsigned count)
  {
    DEBUG_MSG (SUBSET_REPACK, nullptr,
               "  Shrinking PairPosFormat2 (%u) to [0, %u).",
               split_context.this_index,
               count);
    unsigned old_count = class1Count;
    if (count >= old_count)
      return true;

    graph_t& graph = split_context.c.graph;
    class1Count = count;
    graph.vertices_[split_context.this_index].obj.tail -=
        (old_count - count) * split_context.class1_record_size;

    auto coverage =
        graph.as_mutable_table<Coverage> (split_context.this_index, &this->coverage);
    if (!coverage) return false;

    auto class_def_1 =
        graph.as_mutable_table<ClassDef> (split_context.this_index, &classDef1);
    if (!class_def_1) return false;

    auto klass_map =
    + coverage.table->iter ()
    | hb_map_retains_sorting ([&] (hb_codepoint_t gid) {
      return hb_codepoint_pair_t (gid, class_def_1.table->get_class (gid));
    })
    | hb_filter ([&] (hb_codepoint_t klass) {
      return klass < count;
    }, hb_second)
    ;

    auto new_coverage = + klass_map | hb_map_retains_sorting (hb_first);
    if (!Coverage::make_coverage (split_context.c,
                                  + new_coverage,
                                  coverage.index,
                                  // existing ranges my not be kept, worst case size is a format 1
                                  // coverage table.
                                  4 + new_coverage.len() * 2))
      return false;

    return ClassDef::make_class_def (split_context.c,
                                     + klass_map,
                                     class_def_1.index,
                                     class_def_1.vertex->table_size ());
  }

  hb_hashmap_t<unsigned, unsigned>
  get_all_device_tables (gsubgpos_graph_context_t& c,
                         unsigned this_index) const
  {
    const auto& v = c.graph.vertices_[this_index];
    return v.position_to_index_map ();
  }

  const Coverage* get_coverage (gsubgpos_graph_context_t& c,
                          unsigned this_index) const
  {
    unsigned coverage_id = c.graph.index_for_offset (this_index, &coverage);
    auto& coverage_v = c.graph.vertices_[coverage_id];

    Coverage* coverage_table = (Coverage*) coverage_v.obj.head;
    if (!coverage_table || !coverage_table->sanitize (coverage_v))
      return &Null(Coverage);
    return coverage_table;
  }

  const ClassDef* get_class_def_1 (gsubgpos_graph_context_t& c,
                                   unsigned this_index) const
  {
    unsigned class_def_1_id = c.graph.index_for_offset (this_index, &classDef1);
    auto& class_def_1_v = c.graph.vertices_[class_def_1_id];

    ClassDef* class_def_1_table = (ClassDef*) class_def_1_v.obj.head;
    if (!class_def_1_table || !class_def_1_table->sanitize (class_def_1_v))
      return &Null(ClassDef);
    return class_def_1_table;
  }

  unsigned size_of_value_record_children (gsubgpos_graph_context_t& c,
                                          const hb_hashmap_t<unsigned, unsigned>& device_tables,
                                          const hb_vector_t<unsigned> device_table_indices,
                                          unsigned value_record_index,
                                          hb_set_t& visited)
  {
    unsigned size = 0;
    for (unsigned i : device_table_indices)
    {
      OT::Layout::GPOS_impl::Value* record = &values[value_record_index + i];
      unsigned record_position = ((char*) record) - ((char*) this);
      unsigned* obj_idx;
      if (!device_tables.has (record_position, &obj_idx)) continue;
      size += c.graph.find_subgraph_size (*obj_idx, visited);
    }
    return size;
  }

  unsigned size_of (gsubgpos_graph_context_t& c,
                    unsigned this_index,
                    const void* offset) const
  {
    const unsigned id = c.graph.index_for_offset (this_index, offset);
    return c.graph.vertices_[id].table_size ();
  }
};

struct PairPos : public OT::Layout::GPOS_impl::PairPos
{
  hb_vector_t<unsigned> split_subtables (gsubgpos_graph_context_t& c,
                                         unsigned parent_index,
                                         unsigned this_index)
  {
    switch (u.format) {
    case 1:
      return ((PairPosFormat1*)(&u.format1))->split_subtables (c, parent_index, this_index);
    case 2:
      return ((PairPosFormat2*)(&u.format2))->split_subtables (c, parent_index, this_index);
#ifndef HB_NO_BEYOND_64K
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
    hb_barrier ();

    switch (u.format) {
    case 1:
      return ((PairPosFormat1*)(&u.format1))->sanitize (vertex);
    case 2:
      return ((PairPosFormat2*)(&u.format2))->sanitize (vertex);
#ifndef HB_NO_BEYOND_64K
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
