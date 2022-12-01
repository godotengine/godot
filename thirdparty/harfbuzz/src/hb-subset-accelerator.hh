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

#ifndef HB_SUBSET_ACCELERATOR_HH
#define HB_SUBSET_ACCELERATOR_HH


#include "hb.hh"

#include "hb-map.hh"
#include "hb-set.hh"

struct hb_subset_accelerator_t
{
  static hb_user_data_key_t* user_data_key()
  {
    static hb_user_data_key_t key;
    return &key;
  }

  static hb_subset_accelerator_t* create(const hb_map_t& unicode_to_gid_,
                                         const hb_set_t& unicodes_) {
    hb_subset_accelerator_t* accel =
        (hb_subset_accelerator_t*) hb_malloc (sizeof(hb_subset_accelerator_t));
    new (accel) hb_subset_accelerator_t (unicode_to_gid_, unicodes_);
    return accel;
  }

  static void destroy(void* value) {
    if (!value) return;

    hb_subset_accelerator_t* accel = (hb_subset_accelerator_t*) value;
    accel->~hb_subset_accelerator_t ();
    hb_free (accel);
  }

  hb_subset_accelerator_t(const hb_map_t& unicode_to_gid_,
                          const hb_set_t& unicodes_)
      : unicode_to_gid(unicode_to_gid_), unicodes(unicodes_) {}

  const hb_map_t unicode_to_gid;
  const hb_set_t unicodes;
  // TODO(garretrieger): cumulative glyf checksum map
  // TODO(garretrieger): sanitized table cache.

  bool in_error () const
  {
    return unicode_to_gid.in_error() || unicodes.in_error ();
  }
};


#endif /* HB_SUBSET_ACCELERATOR_HH */
