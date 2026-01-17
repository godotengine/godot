/*
 * Copyright © 2020  Google, Inc.
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

#ifndef HB_REPACKER_HH
#define HB_REPACKER_HH

#include "hb-open-type.hh"
#include "hb-map.hh"
#include "hb-vector.hh"
#include "graph/graph.hh"
#include "graph/gsubgpos-graph.hh"
#include "graph/serialize.hh"

using graph::graph_t;

/*
 * For a detailed writeup on the overflow resolution algorithm see:
 * docs/repacker.md
 */

struct lookup_size_t
{
  unsigned lookup_index;
  size_t size;
  unsigned num_subtables;

  static int cmp (const void* a, const void* b)
  {
    return cmp ((const lookup_size_t*) a,
                (const lookup_size_t*) b);
  }

  static int cmp (const lookup_size_t* a, const lookup_size_t* b)
  {
    double subtables_per_byte_a = (double) a->num_subtables / (double) a->size;
    double subtables_per_byte_b = (double) b->num_subtables / (double) b->size;
    if (subtables_per_byte_a == subtables_per_byte_b) {
      return b->lookup_index - a->lookup_index;
    }

    double cmp = subtables_per_byte_b - subtables_per_byte_a;
    if (cmp < 0) return -1;
    if (cmp > 0) return 1;
    return 0;
  }
};

static inline
bool _presplit_subtables_if_needed (graph::gsubgpos_graph_context_t& ext_context)
{
  // For each lookup this will check the size of subtables and split them as needed
  // so that no subtable is at risk of overflowing. (where we support splitting for
  // that subtable type).
  //
  // TODO(grieger): de-dup newly added nodes as necessary. Probably just want a full de-dup
  //                pass after this processing is done. Not super necessary as splits are
  //                only done where overflow is likely, so de-dup probably will get undone
  //                later anyways.

  // The loop below can modify the contents of ext_context.lookups if new subtables are added
  // to a lookup during a split. So save the initial set of lookup indices so the iteration doesn't
  // risk access free'd memory if ext_context.lookups gets resized.
  hb_set_t lookup_indices(ext_context.lookups.keys ());
  for (unsigned lookup_index : lookup_indices)
  {
    graph::Lookup* lookup = ext_context.lookups.get(lookup_index);
    if (!lookup->split_subtables_if_needed (ext_context, lookup_index))
      return false;
  }

  return true;
}

/*
 * Analyze the lookups in a GSUB/GPOS table and decide if any should be promoted
 * to extension lookups.
 */
static inline
bool _promote_extensions_if_needed (graph::gsubgpos_graph_context_t& ext_context)
{
  // Simple Algorithm (v1, current):
  // 1. Calculate how many bytes each non-extension lookup consumes.
  // 2. Select up to 64k of those to remain as non-extension (greedy, highest subtables per byte first)
  // 3. Promote the rest.
  //
  // Advanced Algorithm (v2, not implemented):
  // 1. Perform connected component analysis using lookups as roots.
  // 2. Compute size of each connected component.
  // 3. Select up to 64k worth of connected components to remain as non-extensions.
  //    (greedy, highest subtables per byte first)
  // 4. Promote the rest.

  // TODO(garretrieger): support extension demotion, then consider all lookups. Requires advanced algo.
  // TODO(garretrieger): also support extension promotion during iterative resolution phase, then
  //                     we can use a less conservative threshold here.
  // TODO(grieger): skip this for the 24 bit case.
  if (!ext_context.lookups) return true;

  unsigned total_lookup_table_sizes = 0;
  hb_vector_t<lookup_size_t> lookup_sizes;
  lookup_sizes.alloc (ext_context.lookups.get_population (), true);

  for (unsigned lookup_index : ext_context.lookups.keys ())
  {
    const auto& lookup_v = ext_context.graph.vertices_[lookup_index];
    total_lookup_table_sizes += lookup_v.table_size ();

    const graph::Lookup* lookup = ext_context.lookups.get(lookup_index);
    hb_set_t visited;
    lookup_sizes.push (lookup_size_t {
        lookup_index,
        ext_context.graph.find_subgraph_size (lookup_index, visited),
        lookup->number_of_subtables (),
      });
  }

  lookup_sizes.qsort ();

  size_t lookup_list_size = ext_context.graph.vertices_[ext_context.lookup_list_index].table_size ();
  size_t l2_l3_size = lookup_list_size + total_lookup_table_sizes; // Lookup List + Lookups
  size_t l3_l4_size = total_lookup_table_sizes; // Lookups + SubTables
  size_t l4_plus_size = 0; // SubTables + their descendants

  // Start by assuming all lookups are using extension subtables, this size will be removed later
  // if it's decided to not make a lookup extension.
  for (auto p : lookup_sizes)
  {
    // TODO(garretrieger): this overestimates the extension subtables size because some extension subtables may be
    //                     reused. However, we can't correct this until we have connected component analysis in place.
    unsigned subtables_size = p.num_subtables * 8;
    l3_l4_size += subtables_size;
    l4_plus_size += subtables_size;
  }

  bool layers_full = false;
  for (auto p : lookup_sizes)
  {
    const graph::Lookup* lookup = ext_context.lookups.get(p.lookup_index);
    if (lookup->is_extension (ext_context.table_tag))
      // already an extension so size is counted by the loop above.
      continue;

    if (!layers_full)
    {
      size_t lookup_size = ext_context.graph.vertices_[p.lookup_index].table_size ();
      hb_set_t visited;
      size_t subtables_size = ext_context.graph.find_subgraph_size (p.lookup_index, visited, 1) - lookup_size;
      size_t remaining_size = p.size - subtables_size - lookup_size;

      l3_l4_size   += subtables_size;
      l3_l4_size   -= p.num_subtables * 8;
      l4_plus_size += subtables_size + remaining_size;

      if (l2_l3_size < (1 << 16)
          && l3_l4_size < (1 << 16)
          && l4_plus_size < (1 << 16)) continue; // this lookup fits within all layers groups

      layers_full = true;
    }

    if (!ext_context.lookups.get(p.lookup_index)->make_extension (ext_context, p.lookup_index))
      return false;
  }

  return true;
}

static inline
bool _try_isolating_subgraphs (const hb_vector_t<graph::overflow_record_t>& overflows,
                               graph_t& sorted_graph)
{
  unsigned space = 0;
  hb_set_t roots_to_isolate;

  for (int i = overflows.length - 1; i >= 0; i--)
  {
    const graph::overflow_record_t& r = overflows[i];

    unsigned root;
    unsigned overflow_space = sorted_graph.space_for (r.parent, &root);
    if (!overflow_space) continue;
    if (sorted_graph.num_roots_for_space (overflow_space) <= 1) continue;

    if (!space) {
      space = overflow_space;
    }

    if (space == overflow_space)
      roots_to_isolate.add(root);
  }

  if (!roots_to_isolate) return false;

  unsigned maximum_to_move = hb_max ((sorted_graph.num_roots_for_space (space) / 2u), 1u);
  if (roots_to_isolate.get_population () > maximum_to_move) {
    // Only move at most half of the roots in a space at a time.
    //
    // Note: this was ported from non-stable ids to stable ids. So to retain the same behaviour
    // with regards to which roots are removed from the set we need to remove them in the topological
    // order, not the object id order.
    int extra = roots_to_isolate.get_population () - maximum_to_move;
    for (unsigned id : sorted_graph.ordering_) {
      if (!extra) break;
      if (roots_to_isolate.has(id)) {
        roots_to_isolate.del(id);
        extra--;
      }
    }
  }

  DEBUG_MSG (SUBSET_REPACK, nullptr,
             "Overflow in space %u (%u roots). Moving %u roots to space %u.",
             space,
             sorted_graph.num_roots_for_space (space),
             roots_to_isolate.get_population (),
             sorted_graph.next_space ());

  sorted_graph.isolate_subgraph (roots_to_isolate);
  sorted_graph.move_to_new_space (roots_to_isolate);

  return true;
}

static inline
bool _resolve_shared_overflow(const hb_vector_t<graph::overflow_record_t>& overflows,
                              int overflow_index,
                              graph_t& sorted_graph)
{
  const graph::overflow_record_t& r = overflows[overflow_index];

  // Find all of the parents in overflowing links that link to this
  // same child node. We will then try duplicating the child node and
  // re-assigning all of these parents to the duplicate.
  hb_set_t parents;
  parents.add(r.parent);
  for (int i = overflow_index - 1; i >= 0; i--) {
    const graph::overflow_record_t& r2 = overflows[i];
    if (r2.child == r.child) {
      parents.add(r2.parent);
    }
  }

  unsigned result = sorted_graph.duplicate(&parents, r.child);
  if (result == (unsigned) -1 && parents.get_population() > 2) {
    // All links to the child are overflowing, so we can't include all
    // in the duplication. Remove one parent from the duplication.
    // Remove the lowest index parent, which will be the closest to the child.
    parents.del(parents.get_min());
    result = sorted_graph.duplicate(&parents, r.child);
  }

  if (result == (unsigned) -1) return false;

  if (parents.get_population() > 1) {
    // If the duplicated node has more than one parent pre-emptively raise it's priority to the maximum.
    // This will place it close to the parents. Node's with only one parent, don't need this as normal overflow
    // resolution will raise priority if needed.
    //
    // Reasoning: most of the parents to this child are likely at the same layer in the graph. Duplicating
    // the child will theoretically allow it to be placed closer to it's parents. However, due to the shortest
    // distance sort by default it's placement will remain in the same layer, thus it will remain in roughly the
    // same position (and distance from parents) as the original child node. The overflow resolution will attempt
    // to move nodes closer, but only for non-shared nodes. Since this node is shared, it will simply be given
    // further duplication which defeats the attempt to duplicate with multiple parents. To fix this we
    // pre-emptively raise priority now which allows the duplicated node to pack into the same layer as it's parents.
    sorted_graph.vertices_[result].give_max_priority();
  }

  return true;
}

static inline
bool _process_overflows (const hb_vector_t<graph::overflow_record_t>& overflows,
                         hb_set_t& priority_bumped_parents,
                         graph_t& sorted_graph)
{
  bool resolution_attempted = false;

  // Try resolving the furthest overflows first.
  for (int i = overflows.length - 1; i >= 0; i--)
  {
    const graph::overflow_record_t& r = overflows[i];
    const auto& child = sorted_graph.vertices_[r.child];
    if (child.is_shared ())
    {
      // The child object is shared, we may be able to eliminate the overflow
      // by duplicating it.
      if (_resolve_shared_overflow(overflows, i, sorted_graph))
        return true;

      // Sometimes we can't duplicate a node which looks shared because it's not actually shared
      // (eg. all links from the same parent) in this case continue on to other resolution options.
    }

    if (child.is_leaf () && !priority_bumped_parents.has (r.parent))
    {
      // This object is too far from it's parent, attempt to move it closer.
      //
      // TODO(garretrieger): initially limiting this to leaf's since they can be
      //                     moved closer with fewer consequences. However, this can
      //                     likely can be used for non-leafs as well.
      // TODO(garretrieger): also try lowering priority of the parent. Make it
      //                     get placed further up in the ordering, closer to it's children.
      //                     this is probably preferable if the total size of the parent object
      //                     is < then the total size of the children (and the parent can be moved).
      //                     Since in that case moving the parent will cause a smaller increase in
      //                     the length of other offsets.
      if (sorted_graph.raise_childrens_priority (r.parent)) {
        priority_bumped_parents.add (r.parent);
        resolution_attempted = true;
      }
      continue;
    }

    // TODO(garretrieger): add additional offset resolution strategies
    // - Promotion to extension lookups.
    // - Table splitting.
  }

  return resolution_attempted;
}

inline bool
hb_resolve_graph_overflows (hb_tag_t table_tag,
                            unsigned max_rounds ,
                            bool always_recalculate_extensions,
                            graph_t& sorted_graph /* IN/OUT */)
{
  DEBUG_MSG (SUBSET_REPACK, nullptr, "Repacking %c%c%c%c.", HB_UNTAG(table_tag));
  sorted_graph.sort_shortest_distance ();
  if (sorted_graph.in_error ())
  {
    DEBUG_MSG (SUBSET_REPACK, nullptr, "Sorted graph in error state after initial sort.");
    return false;
  }

  bool will_overflow = graph::will_overflow (sorted_graph);
  if (!will_overflow)
    return true;

  bool is_gsub_or_gpos = (table_tag == HB_OT_TAG_GPOS ||  table_tag == HB_OT_TAG_GSUB);
  graph::gsubgpos_graph_context_t ext_context (table_tag, sorted_graph);
  if (is_gsub_or_gpos && will_overflow)
  {
    DEBUG_MSG (SUBSET_REPACK, nullptr, "Applying GSUB/GPOS repacking specializations.");
    if (always_recalculate_extensions)
    {
      DEBUG_MSG (SUBSET_REPACK, nullptr, "Splitting subtables if needed.");
      if (!_presplit_subtables_if_needed (ext_context)) {
        DEBUG_MSG (SUBSET_REPACK, nullptr, "Subtable splitting failed.");
        return false;
      }

      DEBUG_MSG (SUBSET_REPACK, nullptr, "Promoting lookups to extensions if needed.");
      if (!_promote_extensions_if_needed (ext_context)) {
        DEBUG_MSG (SUBSET_REPACK, nullptr, "Extensions promotion failed.");
        return false;
      }
    }

    DEBUG_MSG (SUBSET_REPACK, nullptr, "Assigning spaces to 32 bit subgraphs.");
    if (sorted_graph.assign_spaces ())
      sorted_graph.sort_shortest_distance ();
    else
      sorted_graph.sort_shortest_distance_if_needed ();
  }

  unsigned round = 0;
  hb_vector_t<graph::overflow_record_t> overflows;
  // TODO(garretrieger): select a good limit for max rounds.
  while (!sorted_graph.in_error ()
         && graph::will_overflow (sorted_graph, &overflows)
         && round < max_rounds) {
    DEBUG_MSG (SUBSET_REPACK, nullptr, "=== Overflow resolution round %u ===", round);
    print_overflows (sorted_graph, overflows);

    hb_set_t priority_bumped_parents;

    if (!_try_isolating_subgraphs (overflows, sorted_graph))
    {
      // Don't count space isolation towards round limit. Only increment
      // round counter if space isolation made no changes.
      round++;
      if (!_process_overflows (overflows, priority_bumped_parents, sorted_graph))
      {
        DEBUG_MSG (SUBSET_REPACK, nullptr, "No resolution available :(");
        break;
      }
    }

    sorted_graph.sort_shortest_distance ();
  }

  if (sorted_graph.in_error ())
  {
    DEBUG_MSG (SUBSET_REPACK, nullptr, "Sorted graph in error state.");
    return false;
  }

  if (graph::will_overflow (sorted_graph))
  {
    if (is_gsub_or_gpos && !always_recalculate_extensions) {
      // If this a GSUB/GPOS table and we didn't try to extension promotion and table splitting then
      // as a last ditch effort, re-run the repacker with it enabled.
      DEBUG_MSG (SUBSET_REPACK, nullptr, "Failed to find a resolution. Re-running with extension promotion and table splitting enabled.");
      return hb_resolve_graph_overflows (table_tag, max_rounds, true, sorted_graph);
    }

    DEBUG_MSG (SUBSET_REPACK, nullptr, "Offset overflow resolution failed.");
    return false;
  }

  return true;
}

/*
 * Attempts to modify the topological sorting of the provided object graph to
 * eliminate offset overflows in the links between objects of the graph. If a
 * non-overflowing ordering is found the updated graph is serialized it into the
 * provided serialization context.
 *
 * If necessary the structure of the graph may be modified in ways that do not
 * affect the functionality of the graph. For example shared objects may be
 * duplicated.
 *
 * For a detailed writeup describing how the algorithm operates see:
 * docs/repacker.md
 */
template<typename T>
inline hb_blob_t*
hb_resolve_overflows (const T& packed,
                      hb_tag_t table_tag,
                      unsigned max_rounds = 32,
                      bool recalculate_extensions = false) {
  graph_t sorted_graph (packed);
  if (sorted_graph.in_error ())
  {
    // Invalid graph definition.
    return nullptr;
  }

  if (!sorted_graph.is_fully_connected ())
  {
    sorted_graph.print_orphaned_nodes ();
    return nullptr;
  }

  if (sorted_graph.in_error ())
  {
    // Allocations failed somewhere
    DEBUG_MSG (SUBSET_REPACK, nullptr,
               "Graph is in error, likely due to a memory allocation error.");
    return nullptr;
  }

  if (!hb_resolve_graph_overflows (table_tag, max_rounds, recalculate_extensions, sorted_graph))
    return nullptr;

  return graph::serialize (sorted_graph);
}

#endif /* HB_REPACKER_HH */
