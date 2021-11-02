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

#ifndef HB_PRIORITY_QUEUE_HH
#define HB_PRIORITY_QUEUE_HH

#include "hb.hh"
#include "hb-vector.hh"

/*
 * hb_priority_queue_t
 *
 * Priority queue implemented as a binary heap. Supports extract minimum
 * and insert operations.
 */
struct hb_priority_queue_t
{
  HB_DELETE_COPY_ASSIGN (hb_priority_queue_t);
  hb_priority_queue_t ()  { init (); }
  ~hb_priority_queue_t () { fini (); }

 private:
  typedef hb_pair_t<int64_t, unsigned> item_t;
  hb_vector_t<item_t> heap;

 public:
  void init () { heap.init (); }

  void fini () { heap.fini (); }

  void reset () { heap.resize (0); }

  bool in_error () const { return heap.in_error (); }

  void insert (int64_t priority, unsigned value)
  {
    heap.push (item_t (priority, value));
    bubble_up (heap.length - 1);
  }

  item_t pop_minimum ()
  {
    item_t result = heap[0];

    heap[0] = heap[heap.length - 1];
    heap.shrink (heap.length - 1);
    bubble_down (0);

    return result;
  }

  const item_t& minimum ()
  {
    return heap[0];
  }

  bool is_empty () const { return heap.length == 0; }
  explicit operator bool () const { return !is_empty (); }
  unsigned int get_population () const { return heap.length; }

  /* Sink interface. */
  hb_priority_queue_t& operator << (item_t item)
  { insert (item.first, item.second); return *this; }

 private:

  static constexpr unsigned parent (unsigned index)
  {
    return (index - 1) / 2;
  }

  static constexpr unsigned left_child (unsigned index)
  {
    return 2 * index + 1;
  }

  static constexpr unsigned right_child (unsigned index)
  {
    return 2 * index + 2;
  }

  void bubble_down (unsigned index)
  {
    unsigned left = left_child (index);
    unsigned right = right_child (index);

    bool has_left = left < heap.length;
    if (!has_left)
      // If there's no left, then there's also no right.
      return;

    bool has_right = right < heap.length;
    if (heap[index].first <= heap[left].first
        && (!has_right || heap[index].first <= heap[right].first))
      return;

    if (!has_right || heap[left].first < heap[right].first)
    {
      swap (index, left);
      bubble_down (left);
      return;
    }

    swap (index, right);
    bubble_down (right);
  }

  void bubble_up (unsigned index)
  {
    if (index == 0) return;

    unsigned parent_index = parent (index);
    if (heap[parent_index].first <= heap[index].first)
      return;

    swap (index, parent_index);
    bubble_up (parent_index);
  }

  void swap (unsigned a, unsigned b)
  {
    item_t temp = heap[a];
    heap[a] = heap[b];
    heap[b] = temp;
  }
};

#endif /* HB_PRIORITY_QUEUE_HH */
