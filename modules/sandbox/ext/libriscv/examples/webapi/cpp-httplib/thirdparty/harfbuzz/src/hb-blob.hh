/*
 * Copyright © 2009  Red Hat, Inc.
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
 * Red Hat Author(s): Behdad Esfahbod
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_BLOB_HH
#define HB_BLOB_HH

#include "hb.hh"


/*
 * hb_blob_t
 */

struct hb_blob_t
{
  ~hb_blob_t () { destroy_user_data (); }

  void destroy_user_data ()
  {
    if (destroy)
    {
      destroy (user_data);
      user_data = nullptr;
      destroy = nullptr;
    }
  }

  HB_INTERNAL bool try_make_writable ();
  HB_INTERNAL bool try_make_writable_inplace ();
  HB_INTERNAL bool try_make_writable_inplace_unix ();

  hb_bytes_t as_bytes () const { return hb_bytes_t (data, length); }
  template <typename Type>
  const Type* as () const { return as_bytes ().as<Type> (); }

  public:
  hb_object_header_t header;

  const char *data = nullptr;
  unsigned int length = 0;
  hb_memory_mode_t mode = (hb_memory_mode_t) 0;

  void *user_data = nullptr;
  hb_destroy_func_t destroy = nullptr;
};


/*
 * hb_blob_ptr_t
 */

template <typename P>
struct hb_blob_ptr_t
{
  typedef hb_remove_pointer<P> T;

  hb_blob_ptr_t (hb_blob_t *b_ = nullptr) : b (b_) {}
  hb_blob_t * operator = (hb_blob_t *b_) { return b = b_; }
  const T * operator -> () const { return get (); }
  const T & operator * () const  { return *get (); }
  template <typename C> operator const C * () const { return get (); }
  operator const char * () const { return (const char *) get (); }
  const T * get () const { return b->as<T> (); }
  hb_blob_t * get_blob () const { return b.get_raw (); }
  unsigned int get_length () const { return b.get ()->length; }
  void destroy () { hb_blob_destroy (b.get_raw ()); b = nullptr; }

  private:
  hb_nonnull_ptr_t<hb_blob_t> b;
};


#endif /* HB_BLOB_HH */
