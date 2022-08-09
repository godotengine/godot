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

#include "../hb-set.hh"
#include "../hb-priority-queue.hh"
#include "../hb-serialize.hh"

#ifndef GRAPH_GRAPH_HH
#define GRAPH_GRAPH_HH

namespace graph {

/**
 * Represents a serialized table in the form of a graph.
 * Provides methods for modifying and reordering the graph.
 */
struct graph_t
{
  struct vertex_t
  {
    hb_serialize_context_t::object_t obj;
    int64_t distance = 0 ;
    int64_t space = 0 ;
    hb_vector_t<unsigned> parents;
    unsigned start = 0;
    unsigned end = 0;
    unsigned priority = 0;

    friend void swap (vertex_t& a, vertex_t& b)
    {
      hb_swap (a.obj, b.obj);
      hb_swap (a.distance, b.distance);
      hb_swap (a.space, b.space);
      hb_swap (a.parents, b.parents);
      hb_swap (a.start, b.start);
      hb_swap (a.end, b.end);
      hb_swap (a.priority, b.priority);
    }

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

    void remove_real_link (unsigned child_index, const void* offset)
    {
      for (unsigned i = 0; i < obj.real_links.length; i++)
      {
        auto& link = obj.real_links[i];
        if (link.objidx != child_index)
          continue;

        if ((obj.head + link.position) != offset)
          continue;

        obj.real_links.remove (i);
        return;
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

    size_t table_size () const {
      return obj.tail - obj.head;
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

  /*
   * A topological sorting of an object graph. Ordered
   * in reverse serialization order (first object in the
   * serialization is at the end of the list). This matches
   * the 'packed' object stack used internally in the
   * serializer
   */
  template<typename T>
  graph_t (const T& objects)
      : parents_invalid (true),
        distance_invalid (true),
        positions_invalid (true),
        successful (true)
  {
    num_roots_for_space_.push (1);
    bool removed_nil = false;
    vertices_.alloc (objects.length);
    vertices_scratch_.alloc (objects.length);
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
    vertices_.fini ();
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

  const hb_serialize_context_t::object_t& object (unsigned i) const
  {
    return vertices_[i].obj;
  }

  /*
   * Generates a new topological sorting of graph ordered by the shortest
   * distance to each node if positions are marked as invalid.
   */
  void sort_shortest_distance_if_needed ()
  {
    if (!positions_invalid) return;
    sort_shortest_distance ();
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
    hb_vector_t<vertex_t> &sorted_graph = vertices_scratch_;
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

      hb_swap (sorted_graph[new_id], vertices_[next_id]);
      const vertex_t& next = sorted_graph[new_id];

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

    remap_all_obj_indices (id_map, &sorted_graph);
    hb_swap (vertices_, sorted_graph);

    if (!check_success (new_id == -1))
      print_orphaned_nodes ();
  }

  /*
   * Finds the set of nodes (placed into roots) that should be assigned unique spaces.
   * More specifically this looks for the top most 24 bit or 32 bit links in the graph.
   * Some special casing is done that is specific to the layout of GSUB/GPOS tables.
   */
  void find_space_roots (hb_set_t& visited, hb_set_t& roots)
  {
    int root_index = (int) root_idx ();
    for (int i = root_index; i >= 0; i--)
    {
      if (visited.has (i)) continue;

      // Only real links can form 32 bit spaces
      for (auto& l : vertices_[i].obj.real_links)
      {
        if (l.is_signed || l.width < 3)
          continue;

        if (i == root_index && l.width == 3)
          // Ignore 24bit links from the root node, this skips past the single 24bit
          // pointer to the lookup list.
          continue;

        if (l.width == 3)
        {
          // A 24bit offset forms a root, unless there is 32bit offsets somewhere
          // in it's subgraph, then those become the roots instead. This is to make sure
          // that extension subtables beneath a 24bit lookup become the spaces instead
          // of the offset to the lookup.
          hb_set_t sub_roots;
          find_32bit_roots (l.objidx, sub_roots);
          if (sub_roots) {
            for (unsigned sub_root_idx : sub_roots) {
              roots.add (sub_root_idx);
              find_subgraph (sub_root_idx, visited);
            }
            continue;
          }
        }

        roots.add (l.objidx);
        find_subgraph (l.objidx, visited);
      }
    }
  }

  unsigned index_for_offset(unsigned node_idx, const void* offset) const
  {
    const auto& node = object (node_idx);
    if (offset < node.head || offset >= node.tail) return -1;

    for (const auto& link : node.real_links)
    {
      if (offset != node.head + link.position)
        continue;
      return link.objidx;
    }

    return -1;
  }


  /*
   * Assign unique space numbers to each connected subgraph of 24 bit and/or 32 bit offset(s).
   * Currently, this is implemented specifically tailored to the structure of a GPOS/GSUB
   * (including with 24bit offsets) table.
   */
  bool assign_spaces ()
  {
    update_parents ();

    hb_set_t visited;
    hb_set_t roots;
    find_space_roots (visited, roots);

    // Mark everything not in the subgraphs of the roots as visited. This prevents
    // subgraphs from being connected via nodes not in those subgraphs.
    visited.invert ();

    if (!roots) return false;

    while (roots)
    {
      unsigned next = HB_SET_VALUE_INVALID;
      if (unlikely (!check_success (!roots.in_error ()))) break;
      if (!roots.next (&next)) break;

      hb_set_t connected_roots;
      find_connected_nodes (next, roots, visited, connected_roots);
      if (unlikely (!check_success (!connected_roots.in_error ()))) break;

      isolate_subgraph (connected_roots);
      if (unlikely (!check_success (!connected_roots.in_error ()))) break;

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
    hb_map_t subgraph;

    // incoming edges to root_idx should be all 32 bit in length so we don't need to de-dup these
    // set the subgraph incoming edge count to match all of root_idx's incoming edges
    hb_set_t parents;
    for (unsigned root_idx : roots)
    {
      subgraph.set (root_idx, wide_parents (root_idx, parents));
      find_subgraph (root_idx, subgraph);
    }

    unsigned original_root_idx = root_idx ();
    hb_map_t index_map;
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
          const unsigned *v;
          if (index_map.has (node_idx, &v)) return *v;
          return node_idx;
        })
        ;

    remap_obj_indices (index_map, new_subgraph);
    remap_obj_indices (index_map, parents.iter (), true);

    // Update roots set with new indices as needed.
    unsigned next = HB_SET_VALUE_INVALID;
    while (roots.next (&next))
    {
      const unsigned *v;
      if (index_map.has (next, &v))
      {
        roots.del (next);
        roots.add (*v);
      }
    }

    return true;
  }

  void find_subgraph (unsigned node_idx, hb_map_t& subgraph)
  {
    for (const auto& link : vertices_[node_idx].obj.all_links ())
    {
      const unsigned *v;
      if (subgraph.has (link.objidx, &v))
      {
        subgraph.set (link.objidx, *v + 1);
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

  size_t find_subgraph_size (unsigned node_idx, hb_set_t& subgraph, unsigned max_depth = -1)
  {
    if (subgraph.has (node_idx)) return 0;
    subgraph.add (node_idx);

    const auto& o = vertices_[node_idx].obj;
    size_t size = o.tail - o.head;
    if (max_depth == 0)
      return size;

    for (const auto& link : o.all_links ())
      size += find_subgraph_size (link.objidx, subgraph, max_depth - 1);
    return size;
  }

  /*
   * Finds the topmost children of 32bit offsets in the subgraph starting
   * at node_idx. Found indices are placed into 'found'.
   */
  void find_32bit_roots (unsigned node_idx, hb_set_t& found)
  {
    for (const auto& link : vertices_[node_idx].obj.all_links ())
    {
      if (!link.is_signed && link.width == 4) {
        found.add (link.objidx);
        continue;
      }
      find_32bit_roots (link.objidx, found);
    }
  }

  /*
   * Moves the child of old_parent_idx pointed to by old_offset to a new
   * vertex at the new_offset.
   */
  template<typename O>
  void move_child (unsigned old_parent_idx,
                   const O* old_offset,
                   unsigned new_parent_idx,
                   const O* new_offset)
  {
    distance_invalid = true;
    positions_invalid = true;

    auto& old_v = vertices_[old_parent_idx];
    auto& new_v = vertices_[new_parent_idx];

    unsigned child_id = index_for_offset (old_parent_idx,
                                          old_offset);

    auto* new_link = new_v.obj.real_links.push ();
    new_link->width = O::static_size;
    new_link->objidx = child_id;
    new_link->position = (const char*) new_offset - (const char*) new_v.obj.head;

    auto& child = vertices_[child_id];
    child.parents.push (new_parent_idx);

    old_v.remove_real_link (child_id, old_offset);
    child.remove_parent (old_parent_idx);
  }

  /*
   * duplicates all nodes in the subgraph reachable from node_idx. Does not re-assign
   * links. index_map is updated with mappings from old id to new id. If a duplication has already
   * been performed for a given index, then it will be skipped.
   */
  void duplicate_subgraph (unsigned node_idx, hb_map_t& index_map)
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
    hb_swap (vertices_[vertices_.length - 2], *clone);

    // Since the root moved, update the parents arrays of all children on the root.
    for (const auto& l : root ().obj.all_links ())
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
   * Adds a new node to the graph, not connected to anything.
   */
  unsigned new_node (char* head, char* tail)
  {
    positions_invalid = true;
    distance_invalid = true;

    auto* clone = vertices_.push ();
    if (vertices_.in_error ()) {
      return -1;
    }

    clone->obj.head = head;
    clone->obj.tail = tail;
    clone->distance = 0;
    clone->space = 0;

    unsigned clone_idx = vertices_.length - 2;

    // The last object is the root of the graph, so swap back the root to the end.
    // The root's obj idx does change, however since it's root nothing else refers to it.
    // all other obj idx's will be unaffected.
    hb_swap (vertices_[vertices_.length - 2], *clone);

    // Since the root moved, update the parents arrays of all children on the root.
    for (const auto& l : root ().obj.all_links ())
      vertices_[l.objidx].remap_parent (root_idx () - 1, root_idx ());

    return clone_idx;
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

  size_t total_size_in_bytes () const {
    size_t total_size = 0;
    for (unsigned i = 0; i < vertices_.length; i++) {
      size_t size = vertices_[i].obj.tail - vertices_[i].obj.head;
      total_size += size;
    }
    return total_size;
  }


 private:

  /*
   * Returns the numbers of incoming edges that are 24 or 32 bits wide.
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
        if (l.objidx == node_idx
            && (l.width == 3 || l.width == 4)
            && !l.is_signed)
        {
          count++;
          parents.add (p);
        }
      }
    }
    return count;
  }

  bool check_success (bool success)
  { return this->successful && (success || ((void) err_other_error (), false)); }

 public:
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
    // (such as a fibonacci queue) with a fast decrease priority.
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

 private:
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
  void remap_obj_indices (const hb_map_t& id_map,
                          Iterator subgraph,
                          bool only_wide = false)
  {
    if (!id_map) return;
    for (unsigned i : subgraph)
    {
      for (auto& link : vertices_[i].obj.all_links_writer ())
      {
        const unsigned *v;
        if (!id_map.has (link.objidx, &v)) continue;
        if (only_wide && !(link.width == 4 && !link.is_signed)) continue;

        reassign_link (link, i, *v);
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
    if (unlikely (!check_success (!visited.in_error ()))) return;
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
  hb_vector_t<vertex_t> vertices_scratch_;
 private:
  bool parents_invalid;
  bool distance_invalid;
  bool positions_invalid;
  bool successful;
  hb_vector_t<unsigned> num_roots_for_space_;
};

}

#endif  // GRAPH_GRAPH_HH
