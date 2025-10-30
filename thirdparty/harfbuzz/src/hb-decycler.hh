/*
 * Copyright © 2025 Behdad Esfahbod
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
 * Author(s): Behdad Esfahbod
 */

#ifndef HB_DECYCLER_HH
#define HB_DECYCLER_HH

#include "hb.hh"

/*
 * hb_decycler_t is an efficient cycle detector for graph traversal.
 * It's a simple tortoise-and-hare algorithm with a twist: it's
 * designed to detect cycles while traversing a graph in a DFS manner,
 * instead of just a linked list.
 *
 * For Floyd's tortoise and hare algorithm, see:
 * https://en.wikipedia.org/wiki/Cycle_detection#Floyd's_tortoise_and_hare
 *
 * hb_decycler_t is O(n) in the number of nodes in the DFS traversal
 * if there are no cycles. Unlike Floyd's algorithm, hb_decycler_t
 * can be used in a DFS traversal, where the graph is not a simple
 * linked list, but a tree with possible cycles.  Like Floyd's algorithm,
 * it is constant-memory (~three  pointers).
 *
 * The decycler works by creating an implicit linked-list on the stack,
 * of the path from the root to the current node, and apply Floyd's
 * algorithm on that list as it goes.
 *
 * The decycler is malloc-free, and as such, much faster to use than a
 * hb_set_t or hb_map_t equivalent.
 *
 * The decycler detects cycles in the graph *eventually*, not *immediately*.
 * That is, it may not detect a cycle until the cycle is fully traversed,
 * even multiple times. See Floyd's algorithm analysis for details.
 *
 * The implementation saves a pointer storage on the stack by combining
 * this->u.decycler and this->u.next into a union.  This is possible because
 * at any point we only need one of those values. The invariant is that
 * after construction, and before destruction, of a node, the u.decycler
 * field is always valid. The u.next field is only valid when the node is
 * in the traversal path, parent to another node.
 *
 * There are three method's:
 *
 *   - hb_decycler_node_t() constructor: Creates a new node in the traversal.
 *     The constructor takes a reference to the decycler object and inserts
 *     itself as the latest node in the traversal path, by advancing the hare
 *     pointer, and for every other descent, advancing the tortoise pointer.
 *
 *   - ~hb_decycler_node_t() destructor: Restores the decycler object to its
 *      previous state by removing the node from the traversal path.
 *
 *   - bool visit(uintptr_t value): Called on every node in the graph.  Returns
 *     true if the node is not part of a cycle, and false if it is.  The value
 *     parameter is used to detect cycles.  It's the caller's responsibility
 *     to ensure that the value is unique for each node in the graph.
 *     The cycle detection is as simple as comparing the value to the value
 *     held by the tortoise pointer, which is the Floyd's algorithm.
 *
 * For usage examples see test-decycler.cc.
 */

struct hb_decycler_node_t;

struct hb_decycler_t
{
  friend struct hb_decycler_node_t;

  private:
  bool tortoise_awake = false;
  hb_decycler_node_t *tortoise = nullptr;
  hb_decycler_node_t *hare = nullptr;
};

struct hb_decycler_node_t
{
  hb_decycler_node_t (hb_decycler_t &decycler)
  {
    u.decycler = &decycler;

    decycler.tortoise_awake = !decycler.tortoise_awake;

    if (!decycler.tortoise)
    {
      // First node.
      assert (decycler.tortoise_awake);
      assert (!decycler.hare);
      decycler.tortoise = decycler.hare = this;
      return;
    }

    if (decycler.tortoise_awake)
      decycler.tortoise = decycler.tortoise->u.next; // Time to move.

    this->prev = decycler.hare;
    decycler.hare->u.next = this;
    decycler.hare = this;
  }

  ~hb_decycler_node_t ()
  {
    hb_decycler_t &decycler = *u.decycler;

    // Inverse of the constructor.

    assert (decycler.hare == this);
    decycler.hare = prev;
    if (prev)
      prev->u.decycler = &decycler;

    assert (decycler.tortoise);
    if (decycler.tortoise_awake)
      decycler.tortoise = decycler.tortoise->prev;

    decycler.tortoise_awake = !decycler.tortoise_awake;
  }

  bool visit (uintptr_t value_)
  {
    value = value_;

    hb_decycler_t &decycler = *u.decycler;

    if (decycler.tortoise == this)
      return true; // First node; not a cycle.

    if (decycler.tortoise->value == value)
      return false; // Cycle detected.

    return true;
  }

  private:
  union {
    hb_decycler_t *decycler;
    hb_decycler_node_t *next;
  } u = {nullptr};
  hb_decycler_node_t *prev = nullptr;
  uintptr_t value = 0;
};

#endif /* HB_DECYCLER_HH */
