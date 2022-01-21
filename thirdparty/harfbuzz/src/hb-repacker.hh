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

/*
 * For a detailed writeup on the overflow resolution algorithm see:
 * docs/repacker.md
 */

struct graph_t
{
  struct vertex_t
  {
    vertex_t () :
        distance (0),
        space (0),
        parents (),
        start (0),
        end (0),
        priority(0) {}

    void fini () {
      obj.fini ();
      parents.fini ();
    }

    hb_serialize_context_t::object_t obj;
    int64_t distance;
    int64_t space;
    hb_vector_t<unsigned> parents;
    unsigned start;
    unsigned end;
    unsigned priority;

    bool is_shared () const
    {
      return parents.length > 1;
    }

    unsigned incoming_edges () const
    {
      return parents.length;
    }

    void remove_parent (unsigned parent_index)
    {
      for (unsigned i = 0; i < parents.length; i++)
      {
        if (parents[i] != parent_index) continue;
        parents.remove (i);
        break;
      }
    }

    void remap_parents (const hb_vector_t<unsigned>& id_map)
    {
      for (unsigned i = 0; i < parents.length; i++)
        parents[i] = id_map[parents[i]];
    }

    void remap_parent (unsigned old_index, unsigned new_index)
    {
      for (unsigned i = 0; i < parents.length; i++)
      {
        if (parents[i] == old_index)
          parents[i] = new_index;
      }
    }

    bool is_leaf () const
    {
      return !obj.real_links.length && !obj.virtual_links.length;
    }

    bool raise_priority ()
    {
      if (has_max_priority ()) return false;
      priority++;
      return true;
    }

    bool has_max_priority () const {
      return priority >= 3;
    }

    int64_t modified_distance (unsigned order) const
    {
      // TODO(garretrieger): once priority is high enough, should try
      // setting distance = 0 which will force to sort immediately after
      // it's parent where possible.

      int64_t modified_distance =
          hb_min (hb_max(distance + distance_modifier (), 0), 0x7FFFFFFFFFF);
      if (has_max_priority ()) {
        modified_distance = 0;
      }
      return (modified_distance << 18) | (0x003FFFF & order);
    }

    int64_t distance_modifier () const
    {
      if (!priority) return 0;
      int64_t table_size = obj.tail - obj.head;

      if (priority == 1)
        return -table_size / 2;

      return -table_size;
    }
  };

  struct overflow_record_t
  {
    unsigned parent;
    unsigned child;
  };

  /*
   * A topological sorting of an object graph. Ordered
   * in reverse serialization order (first object in the
   * serialization is at the end of the list). This matches
   * the 'packed' object stack used internally in the
   * serializer
   */
  graph_t (const hb_vector_t<hb_serialize_context_t::object_t *>& objects)
      : parents_invalid (true),
        distance_invalid (true),
        positions_invalid (true),
        successful (true)
  {
    num_roots_for_space_.push (1);
    bool removed_nil = false;
    for (unsigned i = 0; i < objects.length; i++)
    {
      // TODO(grieger): check all links point to valid objects.

      // If this graph came from a serialization buffer object 0 is the
      // nil object. We don't need it for our purposes here so drop it.
      if (i == 0 && !objects[i])
      {
        removed_nil = true;
        continue;
      }

      vertex_t* v = vertices_.push ();
      if (check_success (!vertices_.in_error ()))
        v->obj = *objects[i];
      if (!removed_nil) continue;
      // Fix indices to account for removed nil object.
      for (auto& l : v->obj.all_links_writer ()) {
        l.objidx--;
      }
    }
  }

  ~graph_t ()
  {
    vertices_.fini_deep ();
  }

  bool in_error () const
  {
    return !successful ||
        vertices_.in_error () ||
        num_roots_for_space_.in_error ();
  }

  const vertex_t& root () const
  {
    return vertices_[root_idx ()];
  }

  unsigned root_idx () const
  {
    // Object graphs are in reverse order, the first object is at the end
    // of the vector. Since the graph is topologically sorted it's safe to
    // assume the first object has no incoming edges.
    return vertices_.length - 1;
  }

  const hb_serialize_context_t::object_t& object(unsigned i) const
  {
    return vertices_[i].obj;
  }

  /*
   * serialize graph into the provided serialization buffer.
   */
  hb_blob_t* serialize () const
  {
    hb_vector_t<char> buffer;
    size_t size = serialized_length ();
    if (!buffer.alloc (size)) {
      DEBUG_MSG (SUBSET_REPACK, nullptr, "Unable to allocate output buffer.");
      return nullptr;
    }
    hb_serialize_context_t c((void *) buffer, size);

    c.start_serialize<void> ();
    for (unsigned i = 0; i < vertices_.length; i++) {
      c.push ();

      size_t size = vertices_[i].obj.tail - vertices_[i].obj.head;
      char* start = c.allocate_size <char> (size);
      if (!start) {
        DEBUG_MSG (SUBSET_REPACK, nullptr, "Buffer out of space.");
        return nullptr;
      }

      memcpy (start, vertices_[i].obj.head, size);

      // Only real links needs to be serialized.
      for (const auto& link : vertices_[i].obj.real_links)
        serialize_link (link, start, &c);

      // All duplications are already encoded in the graph, so don't
      // enable sharing during packing.
      c.pop_pack (false);
    }
    c.end_serialize ();

    if (c.in_error ()) {
      DEBUG_MSG (SUBSET_REPACK, nullptr, "Error during serialization. Err flag: %d",
                 c.errors);
      return nullptr;
    }

    return c.copy_blob ();
  }

  /*
   * Generates a new topological sorting of graph using Kahn's
   * algorithm: https://en.wikipedia.org/wiki/Topological_sorting#Algorithms
   */
  void sort_kahn ()
  {
    positions_invalid = true;

    if (vertices_.length <= 1) {
      // Graph of 1 or less doesn't need sorting.
      return;
    }

    hb_vector_t<unsigned> queue;
    hb_vector_t<vertex_t> sorted_graph;
    if (unlikely (!check_success (sorted_graph.resize (vertices_.length)))) return;
    hb_vector_t<unsigned> id_map;
    if (unlikely (!check_success (id_map.resize (vertices_.length)))) return;

    hb_vector_t<unsigned> removed_edges;
    if (unlikely (!check_success (removed_edges.resize (vertices_.length)))) return;
    update_parents ();

    queue.push (root_idx ());
    int new_id = vertices_.length - 1;

    while (!queue.in_error () && queue.length)
    {
      unsigned next_id = queue[0];
      queue.remove (0);

      vertex_t& next = vertices_[next_id];
      sorted_graph[new_id] = next;
      id_map[next_id] = new_id--;

      for (const auto& link : next.obj.all_links ()) {
        removed_edges[link.objidx]++;
        if (!(vertices_[link.objidx].incoming_edges () - removed_edges[link.objidx]))
          queue.push (link.objidx);
      }
    }

    check_success (!queue.in_error ());
    check_success (!sorted_graph.in_error ());
    if (!check_success (new_id == -1))
      print_orphaned_nodes ();

    remap_all_obj_indices (id_map, &sorted_graph);

    hb_swap (vertices_, sorted_graph);
    sorted_graph.fini_deep ();
  }

  /*
   * Generates a new topological sorting of graph ordered by the shortest
   * distance to each node.
   */
  void sort_shortest_distance ()
  {
    positions_invalid = true;

    if (vertices_.length <= 1) {
      // Graph of 1 or less doesn't need sorting.
      return;
    }

    update_distances ();

    hb_priority_queue_t queue;
    hb_vector_t<vertex_t> sorted_graph;
    if (unlikely (!check_success (sorted_graph.resize (vertices_.length)))) return;
    hb_vector_t<unsigned> id_map;
    if (unlikely (!check_success (id_map.resize (vertices_.length)))) return;

    hb_vector_t<unsigned> removed_edges;
    if (unlikely (!check_success (removed_edges.resize (vertices_.length)))) return;
    update_parents ();

    queue.insert (root ().modified_distance (0), root_idx ());
    int new_id = root_idx ();
    unsigned order = 1;
    while (!queue.in_error () && !queue.is_empty ())
    {
      unsigned next_id = queue.pop_minimum().second;

      vertex_t& next = vertices_[next_id];
      sorted_graph[new_id] = next;
      id_map[next_id] = new_id--;

      for (const auto& link : next.obj.all_links ()) {
        removed_edges[link.objidx]++;
        if (!(vertices_[link.objidx].incoming_edges () - removed_edges[link.objidx]))
          // Add the order that the links were encountered to the priority.
          // This ensures that ties between priorities objects are broken in a consistent
          // way. More specifically this is set up so that if a set of objects have the same
          // distance they'll be added to the topological order in the order that they are
          // referenced from the parent object.
          queue.insert (vertices_[link.objidx].modified_distance (order++),
                        link.objidx);
      }
    }

    check_success (!queue.in_error ());
    check_success (!sorted_graph.in_error ());
    if (!check_success (new_id == -1))
      print_orphaned_nodes ();

    remap_all_obj_indices (id_map, &sorted_graph);

    hb_swap (vertices_, sorted_graph);
    sorted_graph.fini_deep ();
  }

  /*
   * Assign unique space numbers to each connected subgraph of 32 bit offset(s).
   */
  bool assign_32bit_spaces ()
  {
    unsigned root_index = root_idx ();
    hb_set_t visited;
    hb_set_t roots;
    for (unsigned i = 0; i <= root_index; i++)
    {
      // Only real links can form 32 bit spaces
      for (auto& l : vertices_[i].obj.real_links)
      {
        if (l.width == 4 && !l.is_signed)
        {
          roots.add (l.objidx);
          find_subgraph (l.objidx, visited);
        }
      }
    }

    // Mark everything not in the subgraphs of 32 bit roots as visited.
    // This prevents 32 bit subgraphs from being connected via nodes not in the 32 bit subgraphs.
    visited.invert ();

    if (!roots) return false;

    while (roots)
    {
      unsigned next = HB_SET_VALUE_INVALID;
      if (!roots.next (&next)) break;

      hb_set_t connected_roots;
      find_connected_nodes (next, roots, visited, connected_roots);
      isolate_subgraph (connected_roots);

      unsigned next_space = this->next_space ();
      num_roots_for_space_.push (0);
      for (unsigned root : connected_roots)
      {
        DEBUG_MSG (SUBSET_REPACK, nullptr, "Subgraph %u gets space %u", root, next_space);
        vertices_[root].space = next_space;
        num_roots_for_space_[next_space] = num_roots_for_space_[next_space] + 1;
        distance_invalid = true;
        positions_invalid = true;
      }

      // TODO(grieger): special case for GSUB/GPOS use extension promotions to move 16 bit space
      //                into the 32 bit space as needed, instead of using isolation.
    }

    return true;
  }

  /*
   * Isolates the subgraph of nodes reachable from root. Any links to nodes in the subgraph
   * that originate from outside of the subgraph will be removed by duplicating the linked to
   * object.
   *
   * Indices stored in roots will be updated if any of the roots are duplicated to new indices.
   */
  bool isolate_subgraph (hb_set_t& roots)
  {
    update_parents ();
    hb_hashmap_t<unsigned, unsigned> subgraph;

    // incoming edges to root_idx should be all 32 bit in length so we don't need to de-dup these
    // set the subgraph incoming edge count to match all of root_idx's incoming edges
    hb_set_t parents;
    for (unsigned root_idx : roots)
    {
      subgraph.set (root_idx, wide_parents (root_idx, parents));
      find_subgraph (root_idx, subgraph);
    }

    unsigned original_root_idx = root_idx ();
    hb_hashmap_t<unsigned, unsigned> index_map;
    bool made_changes = false;
    for (auto entry : subgraph.iter ())
    {
      const auto& node = vertices_[entry.first];
      unsigned subgraph_incoming_edges = entry.second;

      if (subgraph_incoming_edges < node.incoming_edges ())
      {
        // Only  de-dup objects with incoming links from outside the subgraph.
        made_changes = true;
        duplicate_subgraph (entry.first, index_map);
      }
    }

    if (!made_changes)
      return false;

    if (original_root_idx != root_idx ()
        && parents.has (original_root_idx))
    {
      // If the root idx has changed since parents was determined, update root idx in parents
      parents.add (root_idx ());
      parents.del (original_root_idx);
    }

    auto new_subgraph =
        + subgraph.keys ()
        | hb_map([&] (unsigned node_idx) {
          if (index_map.has (node_idx)) return index_map[node_idx];
          return node_idx;
        })
        ;

    remap_obj_indices (index_map, new_subgraph);
    remap_obj_indices (index_map, parents.iter (), true);

    // Update roots set with new indices as needed.
    unsigned next = HB_SET_VALUE_INVALID;
    while (roots.next (&next))
    {
      if (index_map.has (next))
      {
        roots.del (next);
        roots.add (index_map[next]);
      }
    }

    return true;
  }

  void find_subgraph (unsigned node_idx, hb_hashmap_t<unsigned, unsigned>& subgraph)
  {
    for (const auto& link : vertices_[node_idx].obj.all_links ())
    {
      if (subgraph.has (link.objidx))
      {
        subgraph.set (link.objidx, subgraph[link.objidx] + 1);
        continue;
      }
      subgraph.set (link.objidx, 1);
      find_subgraph (link.objidx, subgraph);
    }
  }

  void find_subgraph (unsigned node_idx, hb_set_t& subgraph)
  {
    if (subgraph.has (node_idx)) return;
    subgraph.add (node_idx);
    for (const auto& link : vertices_[node_idx].obj.all_links ())
      find_subgraph (link.objidx, subgraph);
  }

  /*
   * duplicates all nodes in the subgraph reachable from node_idx. Does not re-assign
   * links. index_map is updated with mappings from old id to new id. If a duplication has already
   * been performed for a given index, then it will be skipped.
   */
  void duplicate_subgraph (unsigned node_idx, hb_hashmap_t<unsigned, unsigned>& index_map)
  {
    if (index_map.has (node_idx))
      return;

    index_map.set (node_idx, duplicate (node_idx));
    for (const auto& l : object (node_idx).all_links ()) {
      duplicate_subgraph (l.objidx, index_map);
    }
  }

  /*
   * Creates a copy of node_idx and returns it's new index.
   */
  unsigned duplicate (unsigned node_idx)
  {
    positions_invalid = true;
    distance_invalid = true;

    auto* clone = vertices_.push ();
    auto& child = vertices_[node_idx];
    if (vertices_.in_error ()) {
      return -1;
    }

    clone->obj.head = child.obj.head;
    clone->obj.tail = child.obj.tail;
    clone->distance = child.distance;
    clone->space = child.space;
    clone->parents.reset ();

    unsigned clone_idx = vertices_.length - 2;
    for (const auto& l : child.obj.real_links)
    {
      clone->obj.real_links.push (l);
      vertices_[l.objidx].parents.push (clone_idx);
    }
    for (const auto& l : child.obj.virtual_links)
    {
      clone->obj.virtual_links.push (l);
      vertices_[l.objidx].parents.push (clone_idx);
    }

    check_success (!clone->obj.real_links.in_error ());
    check_success (!clone->obj.virtual_links.in_error ());

    // The last object is the root of the graph, so swap back the root to the end.
    // The root's obj idx does change, however since it's root nothing else refers to it.
    // all other obj idx's will be unaffected.
    vertex_t root = vertices_[vertices_.length - 2];
    vertices_[clone_idx] = *clone;
    vertices_[vertices_.length - 1] = root;

    // Since the root moved, update the parents arrays of all children on the root.
    for (const auto& l : root.obj.all_links ())
      vertices_[l.objidx].remap_parent (root_idx () - 1, root_idx ());

    return clone_idx;
  }

  /*
   * Creates a copy of child and re-assigns the link from
   * parent to the clone. The copy is a shallow copy, objects
   * linked from child are not duplicated.
   */
  bool duplicate (unsigned parent_idx, unsigned child_idx)
  {
    update_parents ();

    unsigned links_to_child = 0;
    for (const auto& l : vertices_[parent_idx].obj.all_links ())
    {
      if (l.objidx == child_idx) links_to_child++;
    }

    if (vertices_[child_idx].incoming_edges () <= links_to_child)
    {
      // Can't duplicate this node, doing so would orphan the original one as all remaining links
      // to child are from parent.
      DEBUG_MSG (SUBSET_REPACK, nullptr, "  Not duplicating %d => %d",
                 parent_idx, child_idx);
      return false;
    }

    DEBUG_MSG (SUBSET_REPACK, nullptr, "  Duplicating %d => %d",
               parent_idx, child_idx);

    unsigned clone_idx = duplicate (child_idx);
    if (clone_idx == (unsigned) -1) return false;
    // duplicate shifts the root node idx, so if parent_idx was root update it.
    if (parent_idx == clone_idx) parent_idx++;

    auto& parent = vertices_[parent_idx];
    for (auto& l : parent.obj.all_links_writer ())
    {
      if (l.objidx != child_idx)
        continue;

      reassign_link (l, parent_idx, clone_idx);
    }

    return true;
  }

  /*
   * Raises the sorting priority of all children.
   */
  bool raise_childrens_priority (unsigned parent_idx)
  {
    DEBUG_MSG (SUBSET_REPACK, nullptr, "  Raising priority of all children of %d",
               parent_idx);
    // This operation doesn't change ordering until a sort is run, so no need
    // to invalidate positions. It does not change graph structure so no need
    // to update distances or edge counts.
    auto& parent = vertices_[parent_idx].obj;
    bool made_change = false;
    for (auto& l : parent.all_links_writer ())
      made_change |= vertices_[l.objidx].raise_priority ();
    return made_change;
  }

  /*
   * Will any offsets overflow on graph when it's serialized?
   */
  bool will_overflow (hb_vector_t<overflow_record_t>* overflows = nullptr)
  {
    if (overflows) overflows->resize (0);
    update_positions ();

    for (int parent_idx = vertices_.length - 1; parent_idx >= 0; parent_idx--)
    {
      // Don't need to check virtual links for overflow
      for (const auto& link : vertices_[parent_idx].obj.real_links)
      {
        int64_t offset = compute_offset (parent_idx, link);
        if (is_valid_offset (offset, link))
          continue;

        if (!overflows) return true;

        overflow_record_t r;
        r.parent = parent_idx;
        r.child = link.objidx;
        overflows->push (r);
      }
    }

    if (!overflows) return false;
    return overflows->length;
  }

  void print_orphaned_nodes ()
  {
    if (!DEBUG_ENABLED(SUBSET_REPACK)) return;

    DEBUG_MSG (SUBSET_REPACK, nullptr, "Graph is not fully connected.");
    parents_invalid = true;
    update_parents();

    for (unsigned i = 0; i < root_idx (); i++)
    {
      const auto& v = vertices_[i];
      if (!v.parents)
        DEBUG_MSG (SUBSET_REPACK, nullptr, "Node %u is orphaned.", i);
    }
  }

  void print_overflows (const hb_vector_t<overflow_record_t>& overflows)
  {
    if (!DEBUG_ENABLED(SUBSET_REPACK)) return;

    update_parents ();
    int limit = 10;
    for (const auto& o : overflows)
    {
      if (!limit--) break;
      const auto& parent = vertices_[o.parent];
      const auto& child = vertices_[o.child];
      DEBUG_MSG (SUBSET_REPACK, nullptr,
                 "  overflow from "
                 "%4d (%4d in, %4d out, space %2d) => "
                 "%4d (%4d in, %4d out, space %2d)",
                 o.parent,
                 parent.incoming_edges (),
                 parent.obj.real_links.length + parent.obj.virtual_links.length,
                 space_for (o.parent),
                 o.child,
                 child.incoming_edges (),
                 child.obj.real_links.length + child.obj.virtual_links.length,
                 space_for (o.child));
    }
    if (overflows.length > 10) {
      DEBUG_MSG (SUBSET_REPACK, nullptr, "  ... plus %d more overflows.", overflows.length - 10);
    }
  }

  unsigned num_roots_for_space (unsigned space) const
  {
    return num_roots_for_space_[space];
  }

  unsigned next_space () const
  {
    return num_roots_for_space_.length;
  }

  void move_to_new_space (const hb_set_t& indices)
  {
    num_roots_for_space_.push (0);
    unsigned new_space = num_roots_for_space_.length - 1;

    for (unsigned index : indices) {
      auto& node = vertices_[index];
      num_roots_for_space_[node.space] = num_roots_for_space_[node.space] - 1;
      num_roots_for_space_[new_space] = num_roots_for_space_[new_space] + 1;
      node.space = new_space;
      distance_invalid = true;
      positions_invalid = true;
    }
  }

  unsigned space_for (unsigned index, unsigned* root = nullptr) const
  {
    const auto& node = vertices_[index];
    if (node.space)
    {
      if (root != nullptr)
        *root = index;
      return node.space;
    }

    if (!node.parents)
    {
      if (root)
        *root = index;
      return 0;
    }

    return space_for (node.parents[0], root);
  }

  void err_other_error () { this->successful = false; }

 private:

  size_t serialized_length () const {
    size_t total_size = 0;
    for (unsigned i = 0; i < vertices_.length; i++) {
      size_t size = vertices_[i].obj.tail - vertices_[i].obj.head;
      total_size += size;
    }
    return total_size;
  }

  /*
   * Returns the numbers of incoming edges that are 32bits wide.
   */
  unsigned wide_parents (unsigned node_idx, hb_set_t& parents) const
  {
    unsigned count = 0;
    hb_set_t visited;
    for (unsigned p : vertices_[node_idx].parents)
    {
      if (visited.has (p)) continue;
      visited.add (p);

      // Only real links can be wide
      for (const auto& l : vertices_[p].obj.real_links)
      {
        if (l.objidx == node_idx && l.width == 4 && !l.is_signed)
        {
          count++;
          parents.add (p);
        }
      }
    }
    return count;
  }

  bool check_success (bool success)
  { return this->successful && (success || (err_other_error (), false)); }

  /*
   * Creates a map from objid to # of incoming edges.
   */
  void update_parents ()
  {
    if (!parents_invalid) return;

    for (unsigned i = 0; i < vertices_.length; i++)
      vertices_[i].parents.reset ();

    for (unsigned p = 0; p < vertices_.length; p++)
    {
      for (auto& l : vertices_[p].obj.all_links ())
      {
        vertices_[l.objidx].parents.push (p);
      }
    }

    parents_invalid = false;
  }

  /*
   * compute the serialized start and end positions for each vertex.
   */
  void update_positions ()
  {
    if (!positions_invalid) return;

    unsigned current_pos = 0;
    for (int i = root_idx (); i >= 0; i--)
    {
      auto& v = vertices_[i];
      v.start = current_pos;
      current_pos += v.obj.tail - v.obj.head;
      v.end = current_pos;
    }

    positions_invalid = false;
  }

  /*
   * Finds the distance to each object in the graph
   * from the initial node.
   */
  void update_distances ()
  {
    if (!distance_invalid) return;

    // Uses Dijkstra's algorithm to find all of the shortest distances.
    // https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
    //
    // Implementation Note:
    // Since our priority queue doesn't support fast priority decreases
    // we instead just add new entries into the queue when a priority changes.
    // Redundant ones are filtered out later on by the visited set.
    // According to https://www3.cs.stonybrook.edu/~rezaul/papers/TR-07-54.pdf
    // for practical performance this is faster then using a more advanced queue
    // (such as a fibonaacci queue) with a fast decrease priority.
    for (unsigned i = 0; i < vertices_.length; i++)
    {
      if (i == vertices_.length - 1)
        vertices_[i].distance = 0;
      else
        vertices_[i].distance = hb_int_max (int64_t);
    }

    hb_priority_queue_t queue;
    queue.insert (0, vertices_.length - 1);

    hb_vector_t<bool> visited;
    visited.resize (vertices_.length);

    while (!queue.in_error () && !queue.is_empty ())
    {
      unsigned next_idx = queue.pop_minimum ().second;
      if (visited[next_idx]) continue;
      const auto& next = vertices_[next_idx];
      int64_t next_distance = vertices_[next_idx].distance;
      visited[next_idx] = true;

      for (const auto& link : next.obj.all_links ())
      {
        if (visited[link.objidx]) continue;

        const auto& child = vertices_[link.objidx].obj;
        unsigned link_width = link.width ? link.width : 4; // treat virtual offsets as 32 bits wide
        int64_t child_weight = (child.tail - child.head) +
                               ((int64_t) 1 << (link_width * 8)) * (vertices_[link.objidx].space + 1);
        int64_t child_distance = next_distance + child_weight;

        if (child_distance < vertices_[link.objidx].distance)
        {
          vertices_[link.objidx].distance = child_distance;
          queue.insert (child_distance, link.objidx);
        }
      }
    }

    check_success (!queue.in_error ());
    if (!check_success (queue.is_empty ()))
    {
      print_orphaned_nodes ();
      return;
    }

    distance_invalid = false;
  }

  int64_t compute_offset (
      unsigned parent_idx,
      const hb_serialize_context_t::object_t::link_t& link) const
  {
    const auto& parent = vertices_[parent_idx];
    const auto& child = vertices_[link.objidx];
    int64_t offset = 0;
    switch ((hb_serialize_context_t::whence_t) link.whence) {
      case hb_serialize_context_t::whence_t::Head:
        offset = child.start - parent.start; break;
      case hb_serialize_context_t::whence_t::Tail:
        offset = child.start - parent.end; break;
      case hb_serialize_context_t::whence_t::Absolute:
        offset = child.start; break;
    }

    assert (offset >= link.bias);
    offset -= link.bias;
    return offset;
  }

  bool is_valid_offset (int64_t offset,
                        const hb_serialize_context_t::object_t::link_t& link) const
  {
    if (unlikely (!link.width))
      // Virtual links can't overflow.
      return link.is_signed || offset >= 0;

    if (link.is_signed)
    {
      if (link.width == 4)
        return offset >= -((int64_t) 1 << 31) && offset < ((int64_t) 1 << 31);
      else
        return offset >= -(1 << 15) && offset < (1 << 15);
    }
    else
    {
      if (link.width == 4)
        return offset >= 0 && offset < ((int64_t) 1 << 32);
      else if (link.width == 3)
        return offset >= 0 && offset < ((int32_t) 1 << 24);
      else
        return offset >= 0 && offset < (1 << 16);
    }
  }

  /*
   * Updates a link in the graph to point to a different object. Corrects the
   * parents vector on the previous and new child nodes.
   */
  void reassign_link (hb_serialize_context_t::object_t::link_t& link,
                      unsigned parent_idx,
                      unsigned new_idx)
  {
    unsigned old_idx = link.objidx;
    link.objidx = new_idx;
    vertices_[old_idx].remove_parent (parent_idx);
    vertices_[new_idx].parents.push (parent_idx);
  }

  /*
   * Updates all objidx's in all links using the provided mapping. Corrects incoming edge counts.
   */
  template<typename Iterator, hb_requires (hb_is_iterator (Iterator))>
  void remap_obj_indices (const hb_hashmap_t<unsigned, unsigned>& id_map,
                          Iterator subgraph,
                          bool only_wide = false)
  {
    if (!id_map) return;
    for (unsigned i : subgraph)
    {
      for (auto& link : vertices_[i].obj.all_links_writer ())
      {
        if (!id_map.has (link.objidx)) continue;
        if (only_wide && !(link.width == 4 && !link.is_signed)) continue;

        reassign_link (link, i, id_map[link.objidx]);
      }
    }
  }

  /*
   * Updates all objidx's in all links using the provided mapping.
   */
  void remap_all_obj_indices (const hb_vector_t<unsigned>& id_map,
                              hb_vector_t<vertex_t>* sorted_graph) const
  {
    for (unsigned i = 0; i < sorted_graph->length; i++)
    {
      (*sorted_graph)[i].remap_parents (id_map);
      for (auto& link : (*sorted_graph)[i].obj.all_links_writer ())
      {
        link.objidx = id_map[link.objidx];
      }
    }
  }

  template <typename O> void
  serialize_link_of_type (const hb_serialize_context_t::object_t::link_t& link,
                          char* head,
                          hb_serialize_context_t* c) const
  {
    OT::Offset<O>* offset = reinterpret_cast<OT::Offset<O>*> (head + link.position);
    *offset = 0;
    c->add_link (*offset,
                 // serializer has an extra nil object at the start of the
                 // object array. So all id's are +1 of what our id's are.
                 link.objidx + 1,
                 (hb_serialize_context_t::whence_t) link.whence,
                 link.bias);
  }

  void serialize_link (const hb_serialize_context_t::object_t::link_t& link,
                 char* head,
                 hb_serialize_context_t* c) const
  {
    switch (link.width)
    {
    case 0:
      // Virtual links aren't serialized.
      return;
    case 4:
      if (link.is_signed)
      {
        serialize_link_of_type<OT::HBINT32> (link, head, c);
      } else {
        serialize_link_of_type<OT::HBUINT32> (link, head, c);
      }
      return;
    case 2:
      if (link.is_signed)
      {
        serialize_link_of_type<OT::HBINT16> (link, head, c);
      } else {
        serialize_link_of_type<OT::HBUINT16> (link, head, c);
      }
      return;
    case 3:
      serialize_link_of_type<OT::HBUINT24> (link, head, c);
      return;
    default:
      // Unexpected link width.
      assert (0);
    }
  }

  /*
   * Finds all nodes in targets that are reachable from start_idx, nodes in visited will be skipped.
   * For this search the graph is treated as being undirected.
   *
   * Connected targets will be added to connected and removed from targets. All visited nodes
   * will be added to visited.
   */
  void find_connected_nodes (unsigned start_idx,
                             hb_set_t& targets,
                             hb_set_t& visited,
                             hb_set_t& connected)
  {
    if (visited.has (start_idx)) return;
    visited.add (start_idx);

    if (targets.has (start_idx))
    {
      targets.del (start_idx);
      connected.add (start_idx);
    }

    const auto& v = vertices_[start_idx];

    // Graph is treated as undirected so search children and parents of start_idx
    for (const auto& l : v.obj.all_links ())
      find_connected_nodes (l.objidx, targets, visited, connected);

    for (unsigned p : v.parents)
      find_connected_nodes (p, targets, visited, connected);
  }

 public:
  // TODO(garretrieger): make private, will need to move most of offset overflow code into graph.
  hb_vector_t<vertex_t> vertices_;
 private:
  bool parents_invalid;
  bool distance_invalid;
  bool positions_invalid;
  bool successful;
  hb_vector_t<unsigned> num_roots_for_space_;
};

static bool _try_isolating_subgraphs (const hb_vector_t<graph_t::overflow_record_t>& overflows,
                                      graph_t& sorted_graph)
{
  unsigned space = 0;
  hb_set_t roots_to_isolate;

  for (int i = overflows.length - 1; i >= 0; i--)
  {
    const graph_t::overflow_record_t& r = overflows[i];

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

static bool _process_overflows (const hb_vector_t<graph_t::overflow_record_t>& overflows,
                                hb_set_t& priority_bumped_parents,
                                graph_t& sorted_graph)
{
  bool resolution_attempted = false;

  // Try resolving the furthest overflows first.
  for (int i = overflows.length - 1; i >= 0; i--)
  {
    const graph_t::overflow_record_t& r = overflows[i];
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
inline hb_blob_t*
hb_resolve_overflows (const hb_vector_t<hb_serialize_context_t::object_t *>& packed,
                      hb_tag_t table_tag,
                      unsigned max_rounds = 20) {
  // Kahn sort is ~twice as fast as shortest distance sort and works for many fonts
  // so try it first to save time.
  graph_t sorted_graph (packed);
  sorted_graph.sort_kahn ();
  if (!sorted_graph.will_overflow ())
  {
    return sorted_graph.serialize ();
  }

  sorted_graph.sort_shortest_distance ();

  if ((table_tag == HB_OT_TAG_GPOS
       ||  table_tag == HB_OT_TAG_GSUB)
      && sorted_graph.will_overflow ())
  {
    DEBUG_MSG (SUBSET_REPACK, nullptr, "Assigning spaces to 32 bit subgraphs.");
    if (sorted_graph.assign_32bit_spaces ())
      sorted_graph.sort_shortest_distance ();
  }

  unsigned round = 0;
  hb_vector_t<graph_t::overflow_record_t> overflows;
  // TODO(garretrieger): select a good limit for max rounds.
  while (!sorted_graph.in_error ()
         && sorted_graph.will_overflow (&overflows)
         && round++ < max_rounds) {
    DEBUG_MSG (SUBSET_REPACK, nullptr, "=== Overflow resolution round %d ===", round);
    sorted_graph.print_overflows (overflows);

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

  if (sorted_graph.will_overflow ())
  {
    DEBUG_MSG (SUBSET_REPACK, nullptr, "Offset overflow resolution failed.");
    return nullptr;
  }

  return sorted_graph.serialize ();
}

#endif /* HB_REPACKER_HH */
