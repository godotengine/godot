/*
 * Copyright © 2007,2008,2009,2010  Red Hat, Inc.
 * Copyright © 2012,2018  Google, Inc.
 * Copyright © 2019  Facebook, Inc.
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
 * Red Hat Author(s): Behdad Esfahbod
 * Google Author(s): Behdad Esfahbod
 * Facebook Author(s): Behdad Esfahbod
 */

#ifndef HB_SERIALIZE_HH
#define HB_SERIALIZE_HH

#include "hb.hh"
#include "hb-blob.hh"
#include "hb-map.hh"
#include "hb-pool.hh"


/*
 * Serialize
 */

struct hb_serialize_context_t
{
  typedef unsigned objidx_t;

  enum whence_t {
     Head,	/* Relative to the current object head (default). */
     Tail,	/* Relative to the current object tail after packed. */
     Absolute	/* Absolute: from the start of the serialize buffer. */
   };

  struct object_t
  {
    void fini () { links.fini (); }

    bool operator == (const object_t &o) const
    {
      return (tail - head == o.tail - o.head)
	  && (links.length == o.links.length)
	  && 0 == hb_memcmp (head, o.head, tail - head)
	  && links.as_bytes () == o.links.as_bytes ();
    }
    uint32_t hash () const
    {
      return hb_bytes_t (head, tail - head).hash () ^
	     links.as_bytes ().hash ();
    }

    struct link_t
    {
      bool is_wide: 1;
      bool is_signed: 1;
      unsigned whence: 2;
      unsigned position: 28;
      unsigned bias;
      objidx_t objidx;
    };

    char *head;
    char *tail;
    hb_vector_t<link_t> links;
    object_t *next;
  };

  struct snapshot_t
  {
    char *head;
    char *tail;
    object_t *current; // Just for sanity check
    unsigned num_links;
  };

  snapshot_t snapshot ()
  { return snapshot_t { head, tail, current, current->links.length }; }

  hb_serialize_context_t (void *start_, unsigned int size) :
    start ((char *) start_),
    end (start + size),
    current (nullptr)
  { reset (); }
  ~hb_serialize_context_t () { fini (); }

  void fini ()
  {
    for (object_t *_ : ++hb_iter (packed)) _->fini ();
    packed.fini ();
    this->packed_map.fini ();

    while (current)
    {
      auto *_ = current;
      current = current->next;
      _->fini ();
    }
    object_pool.fini ();
  }

  bool in_error () const { return !this->successful; }

  void reset ()
  {
    this->successful = true;
    this->ran_out_of_room = false;
    this->head = this->start;
    this->tail = this->end;
    this->debug_depth = 0;

    fini ();
    this->packed.push (nullptr);
  }

  bool check_success (bool success)
  { return this->successful && (success || (err_other_error (), false)); }

  template <typename T1, typename T2>
  bool check_equal (T1 &&v1, T2 &&v2)
  { return check_success ((long long) v1 == (long long) v2); }

  template <typename T1, typename T2>
  bool check_assign (T1 &v1, T2 &&v2)
  { return check_equal (v1 = v2, v2); }

  template <typename T> bool propagate_error (T &&obj)
  { return check_success (!hb_deref (obj).in_error ()); }

  template <typename T1, typename... Ts> bool propagate_error (T1 &&o1, Ts&&... os)
  { return propagate_error (hb_forward<T1> (o1)) &&
	   propagate_error (hb_forward<Ts> (os)...); }

  /* To be called around main operation. */
  template <typename Type>
  Type *start_serialize ()
  {
    DEBUG_MSG_LEVEL (SERIALIZE, this->start, 0, +1,
		     "start [%p..%p] (%lu bytes)",
		     this->start, this->end,
		     (unsigned long) (this->end - this->start));

    assert (!current);
    return push<Type> ();
  }
  void end_serialize ()
  {
    DEBUG_MSG_LEVEL (SERIALIZE, this->start, 0, -1,
		     "end [%p..%p] serialized %u bytes; %s",
		     this->start, this->end,
		     (unsigned) (this->head - this->start),
		     this->successful ? "successful" : "UNSUCCESSFUL");

    propagate_error (packed, packed_map);

    if (unlikely (!current)) return;
    if (unlikely (in_error())) return;

    assert (!current->next);

    /* Only "pack" if there exist other objects... Otherwise, don't bother.
     * Saves a move. */
    if (packed.length <= 1)
      return;

    pop_pack (false);

    resolve_links ();
  }

  template <typename Type = void>
  Type *push ()
  {
    if (unlikely (in_error ())) return start_embed<Type> ();

    object_t *obj = object_pool.alloc ();
    if (unlikely (!obj))
      check_success (false);
    else
    {
      obj->head = head;
      obj->tail = tail;
      obj->next = current;
      current = obj;
    }
    return start_embed<Type> ();
  }
  void pop_discard ()
  {
    object_t *obj = current;
    if (unlikely (!obj)) return;
    if (unlikely (in_error())) return;

    current = current->next;
    revert (obj->head, obj->tail);
    obj->fini ();
    object_pool.free (obj);
  }

  /* Set share to false when an object is unlikely sharable with others
   * so not worth an attempt, or a contiguous table is serialized as
   * multiple consecutive objects in the reverse order so can't be shared.
   */
  objidx_t pop_pack (bool share=true)
  {
    object_t *obj = current;
    if (unlikely (!obj)) return 0;
    if (unlikely (in_error())) return 0;

    current = current->next;
    obj->tail = head;
    obj->next = nullptr;
    unsigned len = obj->tail - obj->head;
    head = obj->head; /* Rewind head. */

    if (!len)
    {
      assert (!obj->links.length);
      return 0;
    }

    objidx_t objidx;
    if (share)
    {
      objidx = packed_map.get (obj);
      if (objidx)
      {
	obj->fini ();
	return objidx;
      }
    }

    tail -= len;
    memmove (tail, obj->head, len);

    obj->head = tail;
    obj->tail = tail + len;

    packed.push (obj);

    if (unlikely (!propagate_error (packed)))
    {
      /* Obj wasn't successfully added to packed, so clean it up otherwise its
       * links will be leaked. When we use constructor/destructors properly, we
       * can remove these. */
      obj->fini ();
      return 0;
    }

    objidx = packed.length - 1;

    if (share) packed_map.set (obj, objidx);
    propagate_error (packed_map);

    return objidx;
  }

  void revert (snapshot_t snap)
  {
    if (unlikely (in_error ())) return;
    assert (snap.current == current);
    current->links.shrink (snap.num_links);
    revert (snap.head, snap.tail);
  }

  void revert (char *snap_head,
	       char *snap_tail)
  {
    if (unlikely (in_error ())) return;
    assert (snap_head <= head);
    assert (tail <= snap_tail);
    head = snap_head;
    tail = snap_tail;
    discard_stale_objects ();
  }

  void discard_stale_objects ()
  {
    if (unlikely (in_error ())) return;
    while (packed.length > 1 &&
	   packed.tail ()->head < tail)
    {
      packed_map.del (packed.tail ());
      assert (!packed.tail ()->next);
      packed.tail ()->fini ();
      packed.pop ();
    }
    if (packed.length > 1)
      assert (packed.tail ()->head == tail);
  }

  template <typename T>
  void add_link (T &ofs, objidx_t objidx,
		 whence_t whence = Head,
		 unsigned bias = 0)
  {
    static_assert (sizeof (T) == 2 || sizeof (T) == 4, "");
    if (unlikely (in_error ())) return;

    if (!objidx)
      return;

    assert (current);
    assert (current->head <= (const char *) &ofs);

    auto& link = *current->links.push ();

    link.is_wide = sizeof (T) == 4;
    link.is_signed = hb_is_signed (hb_unwrap_type (T));
    link.whence = (unsigned) whence;
    link.position = (const char *) &ofs - current->head;
    link.bias = bias;
    link.objidx = objidx;
  }

  unsigned to_bias (const void *base) const
  {
    if (unlikely (in_error ())) return 0;
    if (!base) return 0;
    assert (current);
    assert (current->head <= (const char *) base);
    return (const char *) base - current->head;
  }

  void resolve_links ()
  {
    if (unlikely (in_error ())) return;

    assert (!current);
    assert (packed.length > 1);

    for (const object_t* parent : ++hb_iter (packed))
      for (const object_t::link_t &link : parent->links)
      {
	const object_t* child = packed[link.objidx];
	if (unlikely (!child)) { err_other_error(); return; }
	unsigned offset = 0;
	switch ((whence_t) link.whence) {
	case Head:     offset = child->head - parent->head; break;
	case Tail:     offset = child->head - parent->tail; break;
	case Absolute: offset = (head - start) + (child->head - tail); break;
	}

	assert (offset >= link.bias);
	offset -= link.bias;
	if (link.is_signed)
	{
	  if (link.is_wide)
	    assign_offset<int32_t> (parent, link, offset);
	  else
	    assign_offset<int16_t> (parent, link, offset);
	}
	else
	{
	  if (link.is_wide)
	    assign_offset<uint32_t> (parent, link, offset);
	  else
	    assign_offset<uint16_t> (parent, link, offset);
	}
      }
  }

  unsigned int length () const
  {
    if (unlikely (!current)) return 0;
    return this->head - current->head;
  }

  void align (unsigned int alignment)
  {
    unsigned int l = length () % alignment;
    if (l)
      allocate_size<void> (alignment - l);
  }

  template <typename Type = void>
  Type *start_embed (const Type *obj HB_UNUSED = nullptr) const
  { return reinterpret_cast<Type *> (this->head); }
  template <typename Type>
  Type *start_embed (const Type &obj) const
  { return start_embed (hb_addressof (obj)); }

  /* Following two functions exist to allow setting breakpoint on. */
  void err_ran_out_of_room () { this->ran_out_of_room = true; }
  void err_other_error () { this->successful = false; }

  template <typename Type>
  Type *allocate_size (unsigned int size)
  {
    if (unlikely (!this->successful)) return nullptr;

    if (this->tail - this->head < ptrdiff_t (size))
    {
      err_ran_out_of_room ();
      this->successful = false;
      return nullptr;
    }
    memset (this->head, 0, size);
    char *ret = this->head;
    this->head += size;
    return reinterpret_cast<Type *> (ret);
  }

  template <typename Type>
  Type *allocate_min ()
  { return this->allocate_size<Type> (Type::min_size); }

  template <typename Type>
  Type *embed (const Type *obj)
  {
    unsigned int size = obj->get_size ();
    Type *ret = this->allocate_size<Type> (size);
    if (unlikely (!ret)) return nullptr;
    memcpy (ret, obj, size);
    return ret;
  }
  template <typename Type>
  Type *embed (const Type &obj)
  { return embed (hb_addressof (obj)); }

  template <typename Type, typename ...Ts> auto
  _copy (const Type &src, hb_priority<1>, Ts&&... ds) HB_RETURN
  (Type *, src.copy (this, hb_forward<Ts> (ds)...))

  template <typename Type> auto
  _copy (const Type &src, hb_priority<0>) -> decltype (&(hb_declval<Type> () = src))
  {
    Type *ret = this->allocate_size<Type> (sizeof (Type));
    if (unlikely (!ret)) return nullptr;
    *ret = src;
    return ret;
  }

  /* Like embed, but active: calls obj.operator=() or obj.copy() to transfer data
   * instead of memcpy(). */
  template <typename Type, typename ...Ts>
  Type *copy (const Type &src, Ts&&... ds)
  { return _copy (src, hb_prioritize, hb_forward<Ts> (ds)...); }
  template <typename Type, typename ...Ts>
  Type *copy (const Type *src, Ts&&... ds)
  { return copy (*src, hb_forward<Ts> (ds)...); }

  template<typename Iterator,
	   hb_requires (hb_is_iterator (Iterator)),
	   typename ...Ts>
  void copy_all (Iterator it, Ts&&... ds)
  { for (decltype (*it) _ : it) copy (_, hb_forward<Ts> (ds)...); }

  template <typename Type>
  hb_serialize_context_t& operator << (const Type &obj) & { embed (obj); return *this; }

  template <typename Type>
  Type *extend_size (Type *obj, unsigned int size)
  {
    if (unlikely (in_error ())) return nullptr;

    assert (this->start <= (char *) obj);
    assert ((char *) obj <= this->head);
    assert ((char *) obj + size >= this->head);
    if (unlikely (!this->allocate_size<Type> (((char *) obj) + size - this->head))) return nullptr;
    return reinterpret_cast<Type *> (obj);
  }
  template <typename Type>
  Type *extend_size (Type &obj, unsigned int size)
  { return extend_size (hb_addressof (obj), size); }

  template <typename Type>
  Type *extend_min (Type *obj) { return extend_size (obj, obj->min_size); }
  template <typename Type>
  Type *extend_min (Type &obj) { return extend_min (hb_addressof (obj)); }

  template <typename Type, typename ...Ts>
  Type *extend (Type *obj, Ts&&... ds)
  { return extend_size (obj, obj->get_size (hb_forward<Ts> (ds)...)); }
  template <typename Type, typename ...Ts>
  Type *extend (Type &obj, Ts&&... ds)
  { return extend (hb_addressof (obj), hb_forward<Ts> (ds)...); }

  /* Output routines. */
  hb_bytes_t copy_bytes () const
  {
    assert (this->successful);
    /* Copy both items from head side and tail side... */
    unsigned int len = (this->head - this->start)
		     + (this->end  - this->tail);

    char *p = (char *) malloc (len);
    if (unlikely (!p)) return hb_bytes_t ();

    memcpy (p, this->start, this->head - this->start);
    memcpy (p + (this->head - this->start), this->tail, this->end - this->tail);
    return hb_bytes_t (p, len);
  }
  template <typename Type>
  Type *copy () const
  { return reinterpret_cast<Type *> ((char *) copy_bytes ().arrayZ); }
  hb_blob_t *copy_blob () const
  {
    hb_bytes_t b = copy_bytes ();
    return hb_blob_create (b.arrayZ, b.length,
			   HB_MEMORY_MODE_WRITABLE,
			   (char *) b.arrayZ, free);
  }

  private:
  template <typename T>
  void assign_offset (const object_t* parent, const object_t::link_t &link, unsigned offset)
  {
    auto &off = * ((BEInt<T> *) (parent->head + link.position));
    assert (0 == off);
    check_assign (off, offset);
  }

  public: /* TODO Make private. */
  char *start, *head, *tail, *end;
  unsigned int debug_depth;
  bool successful;
  bool ran_out_of_room;

  private:

  /* Object memory pool. */
  hb_pool_t<object_t> object_pool;

  /* Stack of currently under construction objects. */
  object_t *current;

  /* Stack of packed objects.  Object 0 is always nil object. */
  hb_vector_t<object_t *> packed;

  /* Map view of packed objects. */
  hb_hashmap_t<const object_t *, objidx_t, nullptr, 0> packed_map;
};


#endif /* HB_SERIALIZE_HH */
