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

#ifndef HB_BLOB_PRIVATE_HH
#define HB_BLOB_PRIVATE_HH

#include "hb-private.hh"


/*
 * hb_blob_t
 */

struct hb_blob_t
{
  inline void fini_shallow (void)
  {
    destroy_user_data ();
  }

  inline void destroy_user_data (void)
  {
    if (destroy)
    {
      destroy (user_data);
      user_data = nullptr;
      destroy = nullptr;
    }
  }

  HB_INTERNAL bool try_make_writable (void);
  HB_INTERNAL bool try_make_writable_inplace (void);
  HB_INTERNAL bool try_make_writable_inplace_unix (void);

  template <typename Type>
  inline const Type* as (void) const
  {
    return unlikely (!data) ? &Null(Type) : reinterpret_cast<const Type *> (data);
  }

  public:
  hb_object_header_t header;
  ASSERT_POD ();

  bool immutable;

  const char *data;
  unsigned int length;
  hb_memory_mode_t mode;

  void *user_data;
  hb_destroy_func_t destroy;
};
DECLARE_NULL_INSTANCE (hb_blob_t);


#endif /* HB_BLOB_PRIVATE_HH */
