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
#include "hb-multimap.hh"
#include "hb-set.hh"

extern HB_INTERNAL hb_user_data_key_t _hb_subset_accelerator_user_data_key;

namespace CFF {
struct cff_subset_accelerator_t;
}

namespace OT {
struct SubtableUnicodesCache;
};

struct hb_subset_accelerator_t
{
  static hb_user_data_key_t* user_data_key()
  {
    return &_hb_subset_accelerator_user_data_key;
  }

  static hb_subset_accelerator_t* create(const hb_map_t& unicode_to_gid_,
					 const hb_multimap_t gid_to_unicodes_,
					 const hb_set_t& unicodes_,
					 bool has_seac_) {
    hb_subset_accelerator_t* accel =
        (hb_subset_accelerator_t*) hb_malloc (sizeof(hb_subset_accelerator_t));
    new (accel) hb_subset_accelerator_t (unicode_to_gid_, gid_to_unicodes_, unicodes_);
    accel->has_seac = has_seac_;
    return accel;
  }

  static void destroy(void* value) {
    if (!value) return;

    hb_subset_accelerator_t* accel = (hb_subset_accelerator_t*) value;

    if (accel->cff_accelerator && accel->destroy_cff_accelerator)
      accel->destroy_cff_accelerator ((void*) accel->cff_accelerator);

    if (accel->cmap_cache && accel->destroy_cmap_cache)
      accel->destroy_cmap_cache ((void*) accel->cmap_cache);

    accel->~hb_subset_accelerator_t ();
    hb_free (accel);
  }

  hb_subset_accelerator_t (const hb_map_t& unicode_to_gid_,
			   const hb_multimap_t& gid_to_unicodes_,
                          const hb_set_t& unicodes_)
      : unicode_to_gid(unicode_to_gid_), gid_to_unicodes (gid_to_unicodes_), unicodes(unicodes_),
        cmap_cache(nullptr), destroy_cmap_cache(nullptr),
        has_seac(false), cff_accelerator(nullptr), destroy_cff_accelerator(nullptr)
  { sanitized_table_cache_lock.init (); }

  ~hb_subset_accelerator_t ()
  { sanitized_table_cache_lock.fini (); }

  // Generic

  mutable hb_mutex_t sanitized_table_cache_lock;
  mutable hb_hashmap_t<hb_tag_t, hb::unique_ptr<hb_blob_t>> sanitized_table_cache;

  const hb_map_t unicode_to_gid;
  const hb_multimap_t gid_to_unicodes;
  const hb_set_t unicodes;

  // cmap
  const OT::SubtableUnicodesCache* cmap_cache;
  hb_destroy_func_t destroy_cmap_cache;

  // CFF
  bool has_seac;
  const CFF::cff_subset_accelerator_t* cff_accelerator;
  hb_destroy_func_t destroy_cff_accelerator;

  // TODO(garretrieger): cumulative glyf checksum map

  bool in_error () const
  {
    return unicode_to_gid.in_error () ||
	   gid_to_unicodes.in_error () ||
	   unicodes.in_error () ||
	   sanitized_table_cache.in_error ();
  }
};


#endif /* HB_SUBSET_ACCELERATOR_HH */
