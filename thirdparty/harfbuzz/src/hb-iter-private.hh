/*
 * Copyright © 2018  Google, Inc.
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

#ifndef HB_ITER_PRIVATE_HH
#define HB_ITER_PRIVATE_HH

#include "hb-private.hh"


/* Unified iterator object.
 *
 * The goal of this template is to make the same iterator interface
 * available to all types, and make it very easy and compact to use.
 * Iterator objects are small, light-weight, objects that can be
 * copied by value.  If the collection / object being iterated on
 * is writable, then the iterator points to lvalues, otherwise it
 * returns rvalues.
 *
 * The way to declare, initialize, and use iterators, eg.:
 *
 *   Iter<const int *> s (src);
 *   Iter<int *> t (dst);
 *   for (; s && t; s++, t++)
 *     *s = *t;
 */

template <typename T>
struct Iter;

#if 0
template <typename T>
struct Iter
{
  explicit inline Iter (const T &c);
};
#endif

template <typename T>
struct Iter<T *>
{
  /* Type of items. */
  typedef T Value;

  /* Constructors. */
  inline Iter (T *array_, int length_) :
    array (array_), length (MAX (length_, 0)) {}
  template <unsigned int length_>
  explicit inline Iter (T (&array_)[length_]) :
    array (array_), length (length_) {}

  /* Emptiness. */
  explicit_operator inline operator bool (void) const { return bool (length); }

  /* Current item. */
  inline T &operator * (void)
  {
    if (unlikely (!length)) return CrapOrNull(T);
    return *array;
  }
  inline T &operator -> (void)
  {
    return (operator *);
  }

  /* Next. */
  inline Iter<T *> & operator ++ (void)
  {
    if (unlikely (!length)) return *this;
    array++;
    length--;
    return *this;
  }
  /* Might return void, or a copy of pre-increment iterator. */
  inline void operator ++ (int)
  {
    if (unlikely (!length)) return;
    array++;
    length--;
  }

  /* Some iterators might implement len(). */
  inline unsigned int len (void) const { return length; }

  /* Some iterators might implement fast-forward.
   * Only implement it if it's constant-time. */
  inline void operator += (unsigned int n)
  {
    n = MIN (n, length);
    array += n;
    length -= n;
  }

  /* Some iterators might implement random-access.
   * Only implement it if it's constant-time. */
  inline Iter<T *> & operator [] (unsigned int i)
  {
    if (unlikely (i >= length)) return CrapOrNull(T);
    return array[i];
  }

  private:
  T *array;
  unsigned int length;
};

/* XXX Remove
 * Just to test these compile. */
static inline void
m (void)
{
  const int src[10] = {};
  int dst[20];

  Iter<const int *> s (src);
  Iter<const int *> s2 (src, 5);
  Iter<int *> t (dst);

  s2 = s;

  for (; s && t; ++s, ++t)
   {
    *t = *s;
   }
}

#endif /* HB_ITER_PRIVATE_HH */
