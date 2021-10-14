/*
 * Copyright © 2011,2012,2013  Google, Inc.
 * Copyright © 2021  Khaled Hosny
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

#ifndef HB_MS_FEATURE_RANGES_HH
#define HB_MS_FEATURE_RANGES_HH

#include "hb.hh"

typedef struct hb_ms_feature_t {
  uint32_t tag_le;
  uint32_t value;
} hb_ms_feature_t;

typedef struct hb_ms_features_t {
  hb_ms_feature_t *features;
  uint32_t         num_features;
} hb_ms_features_t;

struct hb_ms_active_feature_t {
  hb_ms_feature_t fea;
  unsigned int order;

  HB_INTERNAL static int cmp (const void *pa, const void *pb) {
    const auto *a = (const hb_ms_active_feature_t *) pa;
    const auto *b = (const hb_ms_active_feature_t *) pb;
    return a->fea.tag_le < b->fea.tag_le ? -1 : a->fea.tag_le > b->fea.tag_le ? 1 :
	   a->order < b->order ? -1 : a->order > b->order ? 1 :
	   a->fea.value < b->fea.value ? -1 : a->fea.value > b->fea.value ? 1 :
	   0;
  }
  bool operator== (const hb_ms_active_feature_t *f)
  { return cmp (this, f) == 0; }
};

struct hb_ms_feature_event_t {
  unsigned int index;
  bool start;
  hb_ms_active_feature_t feature;

  HB_INTERNAL static int cmp (const void *pa, const void *pb)
  {
    const auto *a = (const hb_ms_feature_event_t *) pa;
    const auto *b = (const hb_ms_feature_event_t *) pb;
    return a->index < b->index ? -1 : a->index > b->index ? 1 :
	   a->start < b->start ? -1 : a->start > b->start ? 1 :
	   hb_ms_active_feature_t::cmp (&a->feature, &b->feature);
  }
};

struct hb_ms_range_record_t {
  hb_ms_features_t features;
  unsigned int index_first; /* == start */
  unsigned int index_last;  /* == end - 1 */
};

HB_INTERNAL bool
hb_ms_setup_features (const hb_feature_t                *features,
		      unsigned int                       num_features,
		      hb_vector_t<hb_ms_feature_t>      &feature_records, /* OUT */
		      hb_vector_t<hb_ms_range_record_t> &range_records /* OUT */);


HB_INTERNAL void
hb_ms_make_feature_ranges (hb_vector_t<hb_ms_feature_t>      &feature_records,
			   hb_vector_t<hb_ms_range_record_t> &range_records,
			   unsigned int                       chars_offset,
			   unsigned int                       chars_len,
			   uint16_t                          *log_clusters,
			   hb_vector_t<hb_ms_features_t*>    &range_features, /* OUT */
			   hb_vector_t<uint32_t>             &range_counts /* OUT */);

#endif /* HB_MS_FEATURE_RANGES_HH */
