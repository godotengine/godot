/*
 * Copyright © 2017,2018  Google, Inc.
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
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_VECTOR_HH
#define HB_VECTOR_HH

#include "hb.hh"
#include "hb-array.hh"
#include "hb-meta.hh"
#include "hb-null.hh"


template <typename Type,
	  bool sorted=false>
struct hb_vector_t
{
  static constexpr bool realloc_move = true;

  typedef Type item_t;
  static constexpr unsigned item_size = hb_static_size (Type);
  using array_t = typename std::conditional<sorted, hb_sorted_array_t<Type>, hb_array_t<Type>>::type;
  using c_array_t = typename std::conditional<sorted, hb_sorted_array_t<const Type>, hb_array_t<const Type>>::type;

  hb_vector_t () = default;
  hb_vector_t (std::initializer_list<Type> lst) : hb_vector_t ()
  {
    alloc (lst.size (), true);
    for (auto&& item : lst)
      push (item);
  }
  template <typename Iterable,
	    hb_requires (hb_is_iterable (Iterable))>
  hb_vector_t (const Iterable &o) : hb_vector_t ()
  {
    auto iter = hb_iter (o);
    if (iter.is_random_access_iterator || iter.has_fast_len)
      alloc (hb_len (iter), true);
    hb_copy (iter, *this);
  }
  hb_vector_t (const hb_vector_t &o) : hb_vector_t ()
  {
    alloc (o.length, true);
    if (unlikely (in_error ())) return;
    copy_array (o.as_array ());
  }
  hb_vector_t (array_t o) : hb_vector_t ()
  {
    alloc (o.length, true);
    if (unlikely (in_error ())) return;
    copy_array (o);
  }
  hb_vector_t (c_array_t o) : hb_vector_t ()
  {
    alloc (o.length, true);
    if (unlikely (in_error ())) return;
    copy_array (o);
  }
  hb_vector_t (hb_vector_t &&o) noexcept
  {
    allocated = o.allocated;
    length = o.length;
    arrayZ = o.arrayZ;
    o.init ();
  }
  ~hb_vector_t () { fini (); }

  public:
  int allocated = 0; /* < 0 means allocation failed. */
  unsigned int length = 0;
  public:
  Type *arrayZ = nullptr;

  void init ()
  {
    allocated = length = 0;
    arrayZ = nullptr;
  }
  void init0 ()
  {
  }

  void fini ()
  {
    /* We allow a hack to make the vector point to a foreign array
     * by the user. In that case length/arrayZ are non-zero but
     * allocated is zero. Don't free anything. */
    if (allocated)
    {
      shrink_vector (0);
      hb_free (arrayZ);
    }
    init ();
  }

  void reset ()
  {
    if (unlikely (in_error ()))
      reset_error ();
    resize (0);
  }

  friend void swap (hb_vector_t& a, hb_vector_t& b) noexcept
  {
    hb_swap (a.allocated, b.allocated);
    hb_swap (a.length, b.length);
    hb_swap (a.arrayZ, b.arrayZ);
  }

  hb_vector_t& operator = (const hb_vector_t &o)
  {
    reset ();
    alloc (o.length, true);
    if (unlikely (in_error ())) return *this;

    copy_array (o.as_array ());

    return *this;
  }
  hb_vector_t& operator = (hb_vector_t &&o) noexcept
  {
    hb_swap (*this, o);
    return *this;
  }

  hb_bytes_t as_bytes () const
  { return hb_bytes_t ((const char *) arrayZ, get_size ()); }

  bool operator == (const hb_vector_t &o) const { return as_array () == o.as_array (); }
  bool operator != (const hb_vector_t &o) const { return !(*this == o); }
  uint32_t hash () const { return as_array ().hash (); }

  Type& operator [] (int i_)
  {
    unsigned int i = (unsigned int) i_;
    if (unlikely (i >= length))
      return Crap (Type);
    return arrayZ[i];
  }
  const Type& operator [] (int i_) const
  {
    unsigned int i = (unsigned int) i_;
    if (unlikely (i >= length))
      return Null (Type);
    return arrayZ[i];
  }

  Type& tail () { return (*this)[length - 1]; }
  const Type& tail () const { return (*this)[length - 1]; }

  explicit operator bool () const { return length; }
  unsigned get_size () const { return length * item_size; }

  /* Sink interface. */
  template <typename T>
  hb_vector_t& operator << (T&& v) { push (std::forward<T> (v)); return *this; }

  array_t   as_array ()       { return hb_array (arrayZ, length); }
  c_array_t as_array () const { return hb_array (arrayZ, length); }

  /* Iterator. */
  typedef c_array_t   iter_t;
  typedef array_t   writer_t;
    iter_t   iter () const { return as_array (); }
  writer_t writer ()       { return as_array (); }
  operator   iter_t () const { return   iter (); }
  operator writer_t ()       { return writer (); }

  /* Faster range-based for loop. */
  Type *begin () const { return arrayZ; }
  Type *end () const { return arrayZ + length; }


  hb_sorted_array_t<Type> as_sorted_array ()
  { return hb_sorted_array (arrayZ, length); }
  hb_sorted_array_t<const Type> as_sorted_array () const
  { return hb_sorted_array (arrayZ, length); }

  template <typename T> explicit operator T * () { return arrayZ; }
  template <typename T> explicit operator const T * () const { return arrayZ; }

  Type * operator  + (unsigned int i) { return arrayZ + i; }
  const Type * operator  + (unsigned int i) const { return arrayZ + i; }

  Type *push ()
  {
    if (unlikely (!resize (length + 1)))
      return std::addressof (Crap (Type));
    return std::addressof (arrayZ[length - 1]);
  }
  template <typename... Args> Type *push (Args&&... args)
  {
    if (unlikely ((int) length >= allocated && !alloc (length + 1)))
      // If push failed to allocate then don't copy v, since this may cause
      // the created copy to leak memory since we won't have stored a
      // reference to it.
      return std::addressof (Crap (Type));

    /* Emplace. */
    Type *p = std::addressof (arrayZ[length++]);
    return new (p) Type (std::forward<Args> (args)...);
  }

  bool in_error () const { return allocated < 0; }
  void set_error ()
  {
    assert (allocated >= 0);
    allocated = -allocated - 1;
  }
  void reset_error ()
  {
    assert (allocated < 0);
    allocated = -(allocated + 1);
  }

  template <typename T = Type,
	    hb_enable_if (hb_is_trivially_copy_assignable(T))>
  Type *
  realloc_vector (unsigned new_allocated, hb_priority<0>)
  {
    if (!new_allocated)
    {
      hb_free (arrayZ);
      return nullptr;
    }
    return (Type *) hb_realloc (arrayZ, new_allocated * sizeof (Type));
  }
  template <typename T = Type,
	    hb_enable_if (!hb_is_trivially_copy_assignable(T))>
  Type *
  realloc_vector (unsigned new_allocated, hb_priority<0>)
  {
    if (!new_allocated)
    {
      hb_free (arrayZ);
      return nullptr;
    }
    Type *new_array = (Type *) hb_malloc (new_allocated * sizeof (Type));
    if (likely (new_array))
    {
      for (unsigned i = 0; i < length; i++)
      {
	new (std::addressof (new_array[i])) Type ();
	new_array[i] = std::move (arrayZ[i]);
	arrayZ[i].~Type ();
      }
      hb_free (arrayZ);
    }
    return new_array;
  }
  /* Specialization for types that can be moved using realloc(). */
  template <typename T = Type,
	    hb_enable_if (T::realloc_move)>
  Type *
  realloc_vector (unsigned new_allocated, hb_priority<1>)
  {
    if (!new_allocated)
    {
      hb_free (arrayZ);
      return nullptr;
    }
    return (Type *) hb_realloc (arrayZ, new_allocated * sizeof (Type));
  }

  template <typename T = Type,
	    hb_enable_if (hb_is_trivially_constructible(T))>
  void
  grow_vector (unsigned size, hb_priority<0>)
  {
    hb_memset (arrayZ + length, 0, (size - length) * sizeof (*arrayZ));
    length = size;
  }
  template <typename T = Type,
	    hb_enable_if (!hb_is_trivially_constructible(T))>
  void
  grow_vector (unsigned size, hb_priority<0>)
  {
    for (; length < size; length++)
      new (std::addressof (arrayZ[length])) Type ();
  }
  /* Specialization for hb_vector_t<hb_{vector,array}_t<U>> to speed up. */
  template <typename T = Type,
	    hb_enable_if (hb_is_same (T, hb_vector_t<typename T::item_t>) ||
			  hb_is_same (T, hb_array_t <typename T::item_t>))>
  void
  grow_vector (unsigned size, hb_priority<1>)
  {
    hb_memset (arrayZ + length, 0, (size - length) * sizeof (*arrayZ));
    length = size;
  }

  template <typename T = Type,
	    hb_enable_if (hb_is_trivially_copyable (T))>
  void
  copy_array (hb_array_t<const Type> other)
  {
    length = other.length;
    if (!HB_OPTIMIZE_SIZE_VAL && sizeof (T) >= sizeof (long long))
      /* This runs faster because of alignment. */
      for (unsigned i = 0; i < length; i++)
	arrayZ[i] = other.arrayZ[i];
    else
       hb_memcpy ((void *) arrayZ, (const void *) other.arrayZ, length * item_size);
  }
  template <typename T = Type,
	    hb_enable_if (!hb_is_trivially_copyable (T) &&
			   std::is_copy_constructible<T>::value)>
  void
  copy_array (hb_array_t<const Type> other)
  {
    length = 0;
    while (length < other.length)
    {
      length++;
      new (std::addressof (arrayZ[length - 1])) Type (other.arrayZ[length - 1]);
    }
  }
  template <typename T = Type,
	    hb_enable_if (!hb_is_trivially_copyable (T) &&
			  !std::is_copy_constructible<T>::value &&
			  std::is_default_constructible<T>::value &&
			  std::is_copy_assignable<T>::value)>
  void
  copy_array (hb_array_t<const Type> other)
  {
    length = 0;
    while (length < other.length)
    {
      length++;
      new (std::addressof (arrayZ[length - 1])) Type ();
      arrayZ[length - 1] = other.arrayZ[length - 1];
    }
  }

  void
  shrink_vector (unsigned size)
  {
    assert (size <= length);
    if (!std::is_trivially_destructible<Type>::value)
    {
      unsigned count = length - size;
      Type *p = arrayZ + length - 1;
      while (count--)
        p--->~Type ();
    }
    length = size;
  }

  void
  shift_down_vector (unsigned i)
  {
    for (; i < length; i++)
      arrayZ[i - 1] = std::move (arrayZ[i]);
  }

  /* Allocate for size but don't adjust length. */
  bool alloc (unsigned int size, bool exact=false)
  {
    if (unlikely (in_error ()))
      return false;

    unsigned int new_allocated;
    if (exact)
    {
      /* If exact was specified, we allow shrinking the storage. */
      size = hb_max (size, length);
      if (size <= (unsigned) allocated &&
	  size >= (unsigned) allocated >> 2)
	return true;

      new_allocated = size;
    }
    else
    {
      if (likely (size <= (unsigned) allocated))
	return true;

      new_allocated = allocated;
      while (size > new_allocated)
	new_allocated += (new_allocated >> 1) + 8;
    }


    /* Reallocate */

    bool overflows =
      (int) in_error () ||
      (new_allocated < size) ||
      hb_unsigned_mul_overflows (new_allocated, sizeof (Type));

    if (unlikely (overflows))
    {
      set_error ();
      return false;
    }

    Type *new_array = realloc_vector (new_allocated, hb_prioritize);

    if (unlikely (new_allocated && !new_array))
    {
      if (new_allocated <= (unsigned) allocated)
        return true; // shrinking failed; it's okay; happens in our fuzzer

      set_error ();
      return false;
    }

    arrayZ = new_array;
    allocated = new_allocated;

    return true;
  }

  bool resize (int size_, bool initialize = true, bool exact = false)
  {
    unsigned int size = size_ < 0 ? 0u : (unsigned int) size_;
    if (!alloc (size, exact))
      return false;

    if (size > length)
    {
      if (initialize)
	grow_vector (size, hb_prioritize);
    }
    else if (size < length)
    {
      if (initialize)
	shrink_vector (size);
    }

    length = size;
    return true;
  }
  bool resize_exact (int size_, bool initialize = true)
  {
    return resize (size_, initialize, true);
  }

  Type pop ()
  {
    if (!length) return Null (Type);
    Type v (std::move (arrayZ[length - 1]));
    arrayZ[length - 1].~Type ();
    length--;
    return v;
  }

  void remove_ordered (unsigned int i)
  {
    if (unlikely (i >= length))
      return;
    shift_down_vector (i + 1);
    arrayZ[length - 1].~Type ();
    length--;
  }

  template <bool Sorted = sorted,
	    hb_enable_if (!Sorted)>
  void remove_unordered (unsigned int i)
  {
    if (unlikely (i >= length))
      return;
    if (i != length - 1)
      arrayZ[i] = std::move (arrayZ[length - 1]);
    arrayZ[length - 1].~Type ();
    length--;
  }

  void shrink (int size_, bool shrink_memory = true)
  {
    unsigned int size = size_ < 0 ? 0u : (unsigned int) size_;
    if (size >= length)
      return;

    shrink_vector (size);

    if (shrink_memory)
      alloc (size, true); /* To force shrinking memory if needed. */
  }


  /* Sorting API. */
  void qsort (int (*cmp)(const void*, const void*) = Type::cmp)
  { as_array ().qsort (cmp); }

  /* Unsorted search API. */
  template <typename T>
  Type *lsearch (const T &x, Type *not_found = nullptr)
  { return as_array ().lsearch (x, not_found); }
  template <typename T>
  const Type *lsearch (const T &x, const Type *not_found = nullptr) const
  { return as_array ().lsearch (x, not_found); }
  template <typename T>
  bool lfind (const T &x, unsigned *pos = nullptr) const
  { return as_array ().lfind (x, pos); }

  /* Sorted search API. */
  template <typename T,
	    bool Sorted=sorted, hb_enable_if (Sorted)>
  Type *bsearch (const T &x, Type *not_found = nullptr)
  { return as_array ().bsearch (x, not_found); }
  template <typename T,
	    bool Sorted=sorted, hb_enable_if (Sorted)>
  const Type *bsearch (const T &x, const Type *not_found = nullptr) const
  { return as_array ().bsearch (x, not_found); }
  template <typename T,
	    bool Sorted=sorted, hb_enable_if (Sorted)>
  bool bfind (const T &x, unsigned int *i = nullptr,
	      hb_not_found_t not_found = HB_NOT_FOUND_DONT_STORE,
	      unsigned int to_store = (unsigned int) -1) const
  { return as_array ().bfind (x, i, not_found, to_store); }
};

template <typename Type>
using hb_sorted_vector_t = hb_vector_t<Type, true>;

#endif /* HB_VECTOR_HH */
