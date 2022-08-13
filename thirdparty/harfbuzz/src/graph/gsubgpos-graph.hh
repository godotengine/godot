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
#include "../OT/Layout/GSUB/ExtensionSubst.hh"
#include "gsubgpos-context.hh"
#include "pairpos-graph.hh"

#ifndef GRAPH_GSUBGPOS_GRAPH_HH
#define GRAPH_GSUBGPOS_GRAPH_HH

namespace graph {

struct Lookup;

template<typename T>
struct ExtensionFormat1 : public OT::ExtensionFormat1<T>
{
  void reset(unsigned type)
  {
    this->format = 1;
    this->extensionLookupType = type;
    this->extensionOffset = 0;
  }

  bool sanitize (graph_t::vertex_t& vertex) const
  {
    int64_t vertex_len = vertex.obj.tail - vertex.obj.head;
    return vertex_len >= OT::ExtensionFormat1<T>::static_size;
  }

  unsigned get_lookup_type () const
  {
    return this->extensionLookupType;
  }

  unsigned get_subtable_index (graph_t& graph, unsigned this_index) const
  {
    return graph.index_for_offset (this_index, &this->extensionOffset);
  }
};

struct Lookup : public OT::Lookup
{
  unsigned number_of_subtables () const
  {
    return subTable.len;
  }

  bool sanitize (graph_t::vertex_t& vertex) const
  {
    int64_t vertex_len = vertex.obj.tail - vertex.obj.head;
    if (vertex_len < OT::Lookup::min_size) return false;
    return vertex_len >= this->get_size ();
  }

  bool is_extension (hb_tag_t table_tag) const
  {
    return lookupType == extension_type (table_tag);
  }

  bool make_extension (gsubgpos_graph_context_t& c,
                       unsigned this_index)
  {
    unsigned type = lookupType;
    unsigned ext_type = extension_type (c.table_tag);
    if (!ext_type || is_extension (c.table_tag))
    {
      // NOOP
      return true;
    }

    DEBUG_MSG (SUBSET_REPACK, nullptr,
               "Promoting lookup type %u (obj %u) to extension.",
               type,
               this_index);

    for (unsigned i = 0; i < subTable.len; i++)
    {
      unsigned subtable_index = c.graph.index_for_offset (this_index, &subTable[i]);
      if (!make_subtable_extension (c,
                                    this_index,
                                    subtable_index))
        return false;
    }

    lookupType = ext_type;
    return true;
  }

  bool split_subtables_if_needed (gsubgpos_graph_context_t& c,
                                  unsigned this_index)
  {
    unsigned type = lookupType;
    bool is_ext = is_extension (c.table_tag);

    if (c.table_tag != HB_OT_TAG_GPOS)
      return true;

    if (!is_ext && type != OT::Layout::GPOS_impl::PosLookupSubTable::Type::Pair)
      return true;

    hb_vector_t<unsigned> all_new_subtables;
    for (unsigned i = 0; i < subTable.len; i++)
    {
      unsigned subtable_index = c.graph.index_for_offset (this_index, &subTable[i]);
      if (is_ext) {
        unsigned ext_subtable_index = subtable_index;
        ExtensionFormat1<OT::Layout::GSUB_impl::ExtensionSubst>* extension =
            (ExtensionFormat1<OT::Layout::GSUB_impl::ExtensionSubst>*)
            c.graph.object (ext_subtable_index).head;
        if (!extension->sanitize (c.graph.vertices_[ext_subtable_index]))
          continue;

        subtable_index = extension->get_subtable_index (c.graph, ext_subtable_index);
        type = extension->get_lookup_type ();
        if (type != OT::Layout::GPOS_impl::PosLookupSubTable::Type::Pair)
          continue;
      }

      PairPos* pairPos = (PairPos*) c.graph.object (subtable_index).head;
      if (!pairPos->sanitize (c.graph.vertices_[subtable_index])) continue;

      hb_vector_t<unsigned> new_sub_tables = pairPos->split_subtables (c, subtable_index);
      if (new_sub_tables.in_error ()) return false;
      + new_sub_tables.iter() | hb_sink (all_new_subtables);
    }

    if (all_new_subtables)
      add_sub_tables (c, this_index, type, all_new_subtables);

    return true;
  }

  void add_sub_tables (gsubgpos_graph_context_t& c,
                       unsigned this_index,
                       unsigned type,
                       hb_vector_t<unsigned>& subtable_indices)
  {
    bool is_ext = is_extension (c.table_tag);
    auto& v = c.graph.vertices_[this_index];

    size_t new_size = v.table_size ()
                      + subtable_indices.length * OT::Offset16::static_size;
    char* buffer = (char*) hb_calloc (1, new_size);
    c.add_buffer (buffer);
    memcpy (buffer, v.obj.head, v.table_size());

    v.obj.head = buffer;
    v.obj.tail = buffer + new_size;

    Lookup* new_lookup = (Lookup*) buffer;

    new_lookup->subTable.len = subTable.len + subtable_indices.length;
    unsigned offset_index = subTable.len;
    for (unsigned subtable_id : subtable_indices)
    {
      if (is_ext)
      {
        unsigned ext_id = create_extension_subtable (c, subtable_id, type);
        c.graph.vertices_[subtable_id].parents.push (ext_id);
        subtable_id = ext_id;
      }

      auto* link = v.obj.real_links.push ();
      link->width = 2;
      link->objidx = subtable_id;
      link->position = (char*) &new_lookup->subTable[offset_index++] -
                       (char*) new_lookup;
      c.graph.vertices_[subtable_id].parents.push (this_index);
    }

    // The head location of the lookup has changed, invalidating the lookups map entry
    // in the context. Update the map.
    c.lookups.set (this_index, new_lookup);
  }

  unsigned create_extension_subtable (gsubgpos_graph_context_t& c,
                                      unsigned subtable_index,
                                      unsigned type)
  {
    unsigned extension_size = OT::ExtensionFormat1<OT::Layout::GSUB_impl::ExtensionSubst>::static_size;

    unsigned ext_index = c.create_node (extension_size);
    if (ext_index == (unsigned) -1)
      return -1;

    auto& ext_vertex = c.graph.vertices_[ext_index];
    ExtensionFormat1<OT::Layout::GSUB_impl::ExtensionSubst>* extension =
        (ExtensionFormat1<OT::Layout::GSUB_impl::ExtensionSubst>*) ext_vertex.obj.head;
    extension->reset (type);

    // Make extension point at the subtable.
    auto* l = ext_vertex.obj.real_links.push ();

    l->width = 4;
    l->objidx = subtable_index;
    l->position = 4;

    return ext_index;
  }

  bool make_subtable_extension (gsubgpos_graph_context_t& c,
                                unsigned lookup_index,
                                unsigned subtable_index)
  {
    unsigned type = lookupType;

    unsigned ext_index = create_extension_subtable(c, subtable_index, type);
    if (ext_index == (unsigned) -1)
      return false;

    auto& lookup_vertex = c.graph.vertices_[lookup_index];
    for (auto& l : lookup_vertex.obj.real_links.writer ())
    {
      if (l.objidx == subtable_index)
        // Change lookup to point at the extension.
        l.objidx = ext_index;
    }

    // Make extension point at the subtable.
    auto& ext_vertex = c.graph.vertices_[ext_index];
    auto& subtable_vertex = c.graph.vertices_[subtable_index];
    ext_vertex.parents.push (lookup_index);
    subtable_vertex.remap_parent (lookup_index, ext_index);

    return true;
  }

 private:
  unsigned extension_type (hb_tag_t table_tag) const
  {
    switch (table_tag)
    {
    case HB_OT_TAG_GPOS: return 9;
    case HB_OT_TAG_GSUB: return 7;
    default: return 0;
    }
  }
};

template <typename T>
struct LookupList : public OT::LookupList<T>
{
  bool sanitize (const graph_t::vertex_t& vertex) const
  {
    int64_t vertex_len = vertex.obj.tail - vertex.obj.head;
    if (vertex_len < OT::LookupList<T>::min_size) return false;
    return vertex_len >= OT::LookupList<T>::item_size * this->len;
  }
};

struct GSTAR : public OT::GSUBGPOS
{
  static GSTAR* graph_to_gstar (graph_t& graph)
  {
    const auto& r = graph.root ();

    GSTAR* gstar = (GSTAR*) r.obj.head;
    if (!gstar->sanitize (r))
      return nullptr;

    return gstar;
  }

  const void* get_lookup_list_field_offset () const
  {
    switch (u.version.major) {
    case 1: return u.version1.get_lookup_list_offset ();
#ifndef HB_NO_BORING_EXPANSION
    case 2: return u.version2.get_lookup_list_offset ();
#endif
    default: return 0;
    }
  }

  bool sanitize (const graph_t::vertex_t& vertex)
  {
    int64_t len = vertex.obj.tail - vertex.obj.head;
    if (len < OT::GSUBGPOS::min_size) return false;
    return len >= get_size ();
  }

  void find_lookups (graph_t& graph,
                     hb_hashmap_t<unsigned, Lookup*>& lookups /* OUT */)
  {
    switch (u.version.major) {
      case 1: find_lookups<SmallTypes> (graph, lookups); break;
#ifndef HB_NO_BORING_EXPANSION
      case 2: find_lookups<MediumTypes> (graph, lookups); break;
#endif
    }
  }

  unsigned get_lookup_list_index (graph_t& graph)
  {
    return graph.index_for_offset (graph.root_idx (),
                                   get_lookup_list_field_offset());
  }

  template<typename Types>
  void find_lookups (graph_t& graph,
                     hb_hashmap_t<unsigned, Lookup*>& lookups /* OUT */)
  {
    unsigned lookup_list_idx = get_lookup_list_index (graph);

    const LookupList<Types>* lookupList =
        (const LookupList<Types>*) graph.object (lookup_list_idx).head;
    if (!lookupList->sanitize (graph.vertices_[lookup_list_idx]))
      return;

    for (unsigned i = 0; i < lookupList->len; i++)
    {
      unsigned lookup_idx = graph.index_for_offset (lookup_list_idx, &(lookupList->arrayZ[i]));
      Lookup* lookup = (Lookup*) graph.object (lookup_idx).head;
      if (!lookup->sanitize (graph.vertices_[lookup_idx])) continue;
      lookups.set (lookup_idx, lookup);
    }
  }
};




}

#endif  /* GRAPH_GSUBGPOS_GRAPH_HH */
