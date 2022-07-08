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
#include "hb-priority-queue.hh"
#include "hb-serialize.hh"
#include "hb-vector.hh"
#include "graph/graph.hh"
#include "graph/serialize.hh"

using graph::graph_t;

/*
 * For a detailed writeup on the overflow resolution algorithm see:
 * docs/repacker.md
 */

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
    unsigned extra = roots_to_isolate.get_population () - maximum_to_move;
    while (extra--) {
      unsigned root = HB_SET_VALUE_INVALID;
      roots_to_isolate.previous (&root);
      roots_to_isolate.del (root);
    }
  }

  DEBUG_MSG (SUBSET_REPACK, nullptr,
             "Overflow in space %d (%d roots). Moving %d roots to space %d.",
             space,
             sorted_graph.num_roots_for_space (space),
             roots_to_isolate.get_population (),
             sorted_graph.next_space ());

  sorted_graph.isolate_subgraph (roots_to_isolate);
  sorted_graph.move_to_new_space (roots_to_isolate);

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
      if (!sorted_graph.duplicate (r.parent, r.child)) continue;
      return true;
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
                      unsigned max_rounds = 20) {
  graph_t sorted_graph (packed);
  sorted_graph.sort_shortest_distance ();

  bool will_overflow = graph::will_overflow (sorted_graph);
  if (!will_overflow)
  {
    return graph::serialize (sorted_graph);
  }

  if ((table_tag == HB_OT_TAG_GPOS
       ||  table_tag == HB_OT_TAG_GSUB)
      && will_overflow)
  {
    DEBUG_MSG (SUBSET_REPACK, nullptr, "Assigning spaces to 32 bit subgraphs.");
    if (sorted_graph.assign_32bit_spaces ())
      sorted_graph.sort_shortest_distance ();
  }

  unsigned round = 0;
  hb_vector_t<graph::overflow_record_t> overflows;
  // TODO(garretrieger): select a good limit for max rounds.
  while (!sorted_graph.in_error ()
         && graph::will_overflow (sorted_graph, &overflows)
         && round++ < max_rounds) {
    DEBUG_MSG (SUBSET_REPACK, nullptr, "=== Overflow resolution round %d ===", round);
    print_overflows (sorted_graph, overflows);

    hb_set_t priority_bumped_parents;

    if (!_try_isolating_subgraphs (overflows, sorted_graph))
    {
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
    return nullptr;
  }

  if (graph::will_overflow (sorted_graph))
  {
    DEBUG_MSG (SUBSET_REPACK, nullptr, "Offset overflow resolution failed.");
    return nullptr;
  }

  return graph::serialize (sorted_graph);
}

#endif /* HB_REPACKER_HH */
