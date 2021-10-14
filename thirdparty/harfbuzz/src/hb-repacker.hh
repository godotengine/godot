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


struct graph_t
{
  struct vertex_t
  {
    vertex_t () :
        distance (0),
        incoming_edges (0),
        start (0),
        end (0),
        priority(0) {}

    void fini () { obj.fini (); }

    hb_serialize_context_t::object_t obj;
    int64_t distance;
    unsigned incoming_edges;
    unsigned start;
    unsigned end;
    unsigned priority;

    bool is_shared () const
    {
      return incoming_edges > 1;
    }

    bool is_leaf () const
    {
      return !obj.links.length;
    }

    void raise_priority ()
    {
      priority++;
    }

    int64_t modified_distance (unsigned order) const
    {
      // TODO(garretrieger): once priority is high enough, should try
      // setting distance = 0 which will force to sort immediately after
      // it's parent where possible.

      int64_t modified_distance =
          hb_min (hb_max(distance + distance_modifier (), 0), 0x7FFFFFFFFF);
      return (modified_distance << 24) | (0x00FFFFFF & order);
    }

    int64_t distance_modifier () const
    {
      if (!priority) return 0;
      int64_t table_size = obj.tail - obj.head;
      return -(table_size - table_size / (1 << hb_min(priority, 16u)));
    }
  };

  struct overflow_record_t
  {
    unsigned parent;
    const hb_serialize_context_t::object_t::link_t* link;
  };

  struct clone_buffer_t
  {
    clone_buffer_t () : head (nullptr), tail (nullptr) {}

    bool copy (const hb_serialize_context_t::object_t& object)
    {
      fini ();
      unsigned size = object.tail - object.head;
      head = (char*) hb_malloc (size);
      if (!head) return false;

      memcpy (head, object.head, size);
      tail = head + size;
      return true;
    }

    char* head;
    char* tail;

    void fini ()
    {
      if (!head) return;
      hb_free (head);
      head = nullptr;
    }
  };

  /*
   * A topological sorting of an object graph. Ordered
   * in reverse serialization order (first object in the
   * serialization is at the end of the list). This matches
   * the 'packed' object stack used internally in the
   * serializer
   */
  graph_t (const hb_vector_t<hb_serialize_context_t::object_t *>& objects)
      : edge_count_invalid (true),
        distance_invalid (true),
        positions_invalid (true),
        successful (true)
  {
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
      for (unsigned i = 0; i < v->obj.links.length; i++)
        // Fix indices to account for removed nil object.
        v->obj.links[i].objidx--;
    }
  }

  ~graph_t ()
  {
    vertices_.fini_deep ();
    clone_buffers_.fini_deep ();
  }

  bool in_error () const
  {
    return !successful || vertices_.in_error () || clone_buffers_.in_error ();
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
  void serialize (hb_serialize_context_t* c) const
  {
    c->start_serialize<void> ();
    for (unsigned i = 0; i < vertices_.length; i++) {
      c->push ();

      size_t size = vertices_[i].obj.tail - vertices_[i].obj.head;
      char* start = c->allocate_size <char> (size);
      if (!start) return;

      memcpy (start, vertices_[i].obj.head, size);

      for (const auto& link : vertices_[i].obj.links)
        serialize_link (link, start, c);

      // All duplications are already encoded in the graph, so don't
      // enable sharing during packing.
      c->pop_pack (false);
    }
    c->end_serialize ();
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
    hb_vector_t<unsigned> id_map;
    if (unlikely (!check_success (id_map.resize (vertices_.length)))) return;

    hb_vector_t<unsigned> removed_edges;
    if (unlikely (!check_success (removed_edges.resize (vertices_.length)))) return;
    update_incoming_edge_count ();

    queue.push (root_idx ());
    int new_id = vertices_.length - 1;

    while (!queue.in_error () && queue.length)
    {
      unsigned next_id = queue[0];
      queue.remove (0);

      vertex_t& next = vertices_[next_id];
      sorted_graph.push (next);
      id_map[next_id] = new_id--;

      for (const auto& link : next.obj.links) {
        removed_edges[link.objidx]++;
        if (!(vertices_[link.objidx].incoming_edges - removed_edges[link.objidx]))
          queue.push (link.objidx);
      }
    }

    check_success (!queue.in_error ());
    check_success (!sorted_graph.in_error ());
    if (!check_success (new_id == -1))
      DEBUG_MSG (SUBSET_REPACK, nullptr, "Graph is not fully connected.");

    remap_obj_indices (id_map, &sorted_graph);

    sorted_graph.as_array ().reverse ();

    vertices_.fini_deep ();
    vertices_ = sorted_graph;
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
    hb_vector_t<unsigned> id_map;
    if (unlikely (!check_success (id_map.resize (vertices_.length)))) return;

    hb_vector_t<unsigned> removed_edges;
    if (unlikely (!check_success (removed_edges.resize (vertices_.length)))) return;
    update_incoming_edge_count ();

    queue.insert (root ().modified_distance (0), root_idx ());
    int new_id = root_idx ();
    unsigned order = 1;
    while (!queue.in_error () && !queue.is_empty ())
    {
      unsigned next_id = queue.pop_minimum().second;

      vertex_t& next = vertices_[next_id];
      sorted_graph.push (next);
      id_map[next_id] = new_id--;

      for (const auto& link : next.obj.links) {
        removed_edges[link.objidx]++;
        if (!(vertices_[link.objidx].incoming_edges - removed_edges[link.objidx]))
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
      DEBUG_MSG (SUBSET_REPACK, nullptr, "Graph is not fully connected.");

    remap_obj_indices (id_map, &sorted_graph);

    sorted_graph.as_array ().reverse ();

    vertices_.fini_deep ();
    vertices_ = sorted_graph;
    sorted_graph.fini_deep ();
  }

  /*
   * Creates a copy of child and re-assigns the link from
   * parent to the clone. The copy is a shallow copy, objects
   * linked from child are not duplicated.
   */
  void duplicate (unsigned parent_idx, unsigned child_idx)
  {
    DEBUG_MSG (SUBSET_REPACK, nullptr, "  Duplicating %d => %d",
               parent_idx, child_idx);

    positions_invalid = true;

    auto* clone = vertices_.push ();
    auto& child = vertices_[child_idx];
    clone_buffer_t* buffer = clone_buffers_.push ();
    if (vertices_.in_error ()
        || clone_buffers_.in_error ()
        || !check_success (buffer->copy (child.obj))) {
      return;
    }

    clone->obj.head = buffer->head;
    clone->obj.tail = buffer->tail;
    clone->distance = child.distance;

    for (const auto& l : child.obj.links)
      clone->obj.links.push (l);

    check_success (!clone->obj.links.in_error ());

    auto& parent = vertices_[parent_idx];
    unsigned clone_idx = vertices_.length - 2;
    for (unsigned i = 0; i < parent.obj.links.length; i++)
    {
      auto& l = parent.obj.links[i];
      if (l.objidx == child_idx)
      {
        l.objidx = clone_idx;
        clone->incoming_edges++;
        child.incoming_edges--;
      }
    }

    // The last object is the root of the graph, so swap back the root to the end.
    // The root's obj idx does change, however since it's root nothing else refers to it.
    // all other obj idx's will be unaffected.
    vertex_t root = vertices_[vertices_.length - 2];
    vertices_[vertices_.length - 2] = *clone;
    vertices_[vertices_.length - 1] = root;
  }

  /*
   * Raises the sorting priority of all children.
   */
  void raise_childrens_priority (unsigned parent_idx)
  {
    DEBUG_MSG (SUBSET_REPACK, nullptr, "  Raising priority of all children of %d",
               parent_idx);
    // This operation doesn't change ordering until a sort is run, so no need
    // to invalidate positions. It does not change graph structure so no need
    // to update distances or edge counts.
    auto& parent = vertices_[parent_idx].obj;
    for (unsigned i = 0; i < parent.links.length; i++)
      vertices_[parent.links[i].objidx].raise_priority ();
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
      for (const auto& link : vertices_[parent_idx].obj.links)
      {
        int64_t offset = compute_offset (parent_idx, link);
        if (is_valid_offset (offset, link))
          continue;

        if (!overflows) return true;

        overflow_record_t r;
        r.parent = parent_idx;
        r.link = &link;
        overflows->push (r);
      }
    }

    if (!overflows) return false;
    return overflows->length;
  }

  void print_overflows (const hb_vector_t<overflow_record_t>& overflows)
  {
    if (!DEBUG_ENABLED(SUBSET_REPACK)) return;

    update_incoming_edge_count ();
    for (const auto& o : overflows)
    {
      const auto& child = vertices_[o.link->objidx];
      DEBUG_MSG (SUBSET_REPACK, nullptr, "  overflow from %d => %d (%d incoming , %d outgoing)",
                 o.parent,
                 o.link->objidx,
                 child.incoming_edges,
                 child.obj.links.length);
    }
  }

  void err_other_error () { this->successful = false; }

 private:

  bool check_success (bool success)
  { return this->successful && (success || (err_other_error (), false)); }

  /*
   * Creates a map from objid to # of incoming edges.
   */
  void update_incoming_edge_count ()
  {
    if (!edge_count_invalid) return;

    for (unsigned i = 0; i < vertices_.length; i++)
      vertices_[i].incoming_edges = 0;

    for (const vertex_t& v : vertices_)
    {
      for (auto& l : v.obj.links)
      {
        vertices_[l.objidx].incoming_edges++;
      }
    }

    edge_count_invalid = false;
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

    hb_set_t visited;

    while (!queue.in_error () && !queue.is_empty ())
    {
      unsigned next_idx = queue.pop_minimum ().second;
      if (visited.has (next_idx)) continue;
      const auto& next = vertices_[next_idx];
      int64_t next_distance = vertices_[next_idx].distance;
      visited.add (next_idx);

      for (const auto& link : next.obj.links)
      {
        if (visited.has (link.objidx)) continue;

        const auto& child = vertices_[link.objidx].obj;
        int64_t child_weight = child.tail - child.head +
                               ((int64_t) 1 << (link.width * 8));
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
      DEBUG_MSG (SUBSET_REPACK, nullptr, "Graph is not fully connected.");
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
   * Updates all objidx's in all links using the provided mapping.
   */
  void remap_obj_indices (const hb_vector_t<unsigned>& id_map,
                          hb_vector_t<vertex_t>* sorted_graph) const
  {
    for (unsigned i = 0; i < sorted_graph->length; i++)
    {
      for (unsigned j = 0; j < (*sorted_graph)[i].obj.links.length; j++)
      {
        auto& link = (*sorted_graph)[i].obj.links[j];
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

 public:
  // TODO(garretrieger): make private, will need to move most of offset overflow code into graph.
  hb_vector_t<vertex_t> vertices_;
 private:
  hb_vector_t<clone_buffer_t> clone_buffers_;
  bool edge_count_invalid;
  bool distance_invalid;
  bool positions_invalid;
  bool successful;
};


/*
 * Attempts to modify the topological sorting of the provided object graph to
 * eliminate offset overflows in the links between objects of the graph. If a
 * non-overflowing ordering is found the updated graph is serialized it into the
 * provided serialization context.
 *
 * If necessary the structure of the graph may be modified in ways that do not
 * affect the functionality of the graph. For example shared objects may be
 * duplicated.
 */
inline void
hb_resolve_overflows (const hb_vector_t<hb_serialize_context_t::object_t *>& packed,
                      hb_serialize_context_t* c) {
  // Kahn sort is ~twice as fast as shortest distance sort and works for many fonts
  // so try it first to save time.
  graph_t sorted_graph (packed);
  sorted_graph.sort_kahn ();
  if (!sorted_graph.will_overflow ())
  {
    sorted_graph.serialize (c);
    return;
  }

  sorted_graph.sort_shortest_distance ();

  unsigned round = 0;
  hb_vector_t<graph_t::overflow_record_t> overflows;
  // TODO(garretrieger): select a good limit for max rounds.
  while (!sorted_graph.in_error ()
         && sorted_graph.will_overflow (&overflows)
         && round++ < 10) {
    DEBUG_MSG (SUBSET_REPACK, nullptr, "=== Over flow resolution round %d ===", round);
    sorted_graph.print_overflows (overflows);

    bool resolution_attempted = false;
    hb_set_t priority_bumped_parents;
    // Try resolving the furthest overflows first.
    for (int i = overflows.length - 1; i >= 0; i--)
    {
      const graph_t::overflow_record_t& r = overflows[i];
      const auto& child = sorted_graph.vertices_[r.link->objidx];
      if (child.is_shared ())
      {
        // The child object is shared, we may be able to eliminate the overflow
        // by duplicating it.
        sorted_graph.duplicate (r.parent, r.link->objidx);
        resolution_attempted = true;

        // Stop processing overflows for this round so that object order can be
        // updated to account for the newly added object.
        break;
      }

      if (child.is_leaf () && !priority_bumped_parents.has (r.parent))
      {
        // This object is too far from it's parent, attempt to move it closer.
        //
        // TODO(garretrieger): initially limiting this to leaf's since they can be
        //                     moved closer with fewer consequences. However, this can
        //                     likely can be used for non-leafs as well.
        // TODO(garretrieger): add a maximum priority, don't try to raise past this.
        // TODO(garretrieger): also try lowering priority of the parent. Make it
        //                     get placed further up in the ordering, closer to it's children.
        //                     this is probably preferable if the total size of the parent object
        //                     is < then the total size of the children (and the parent can be moved).
        //                     Since in that case moving the parent will cause a smaller increase in
        //                     the length of other offsets.
        sorted_graph.raise_childrens_priority (r.parent);
        priority_bumped_parents.add (r.parent);
        resolution_attempted = true;
        continue;
      }

      // TODO(garretrieger): add additional offset resolution strategies
      // - Promotion to extension lookups.
      // - Table splitting.
    }

    if (resolution_attempted)
    {
      sorted_graph.sort_shortest_distance ();
      continue;
    }

    DEBUG_MSG (SUBSET_REPACK, nullptr, "No resolution available :(");
    c->err (HB_SERIALIZE_ERROR_OFFSET_OVERFLOW);
    return;
  }

  if (sorted_graph.in_error ())
  {
    c->err (HB_SERIALIZE_ERROR_OTHER);
    return;
  }
  sorted_graph.serialize (c);
}


#endif /* HB_REPACKER_HH */
