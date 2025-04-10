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

#ifndef HB_AAT_MAP_HH
#define HB_AAT_MAP_HH

#include "hb.hh"


struct hb_aat_map_t
{
  friend struct hb_aat_map_builder_t;

  public:
  struct range_flags_t
  {
    hb_mask_t flags;
    unsigned cluster_first;
    unsigned cluster_last; // end - 1
  };

  public:
  hb_vector_t<hb_sorted_vector_t<range_flags_t>> chain_flags;
};

struct hb_aat_map_builder_t
{
  public:

  HB_INTERNAL hb_aat_map_builder_t (hb_face_t *face_,
				    const hb_segment_properties_t props_) :
				      face (face_),
				      props (props_) {}

  HB_INTERNAL void add_feature (const hb_feature_t &feature);

  HB_INTERNAL void compile (hb_aat_map_t  &m);

  public:
  struct feature_info_t
  {
    hb_aat_layout_feature_type_t  type;
    hb_aat_layout_feature_selector_t  setting;
    bool is_exclusive;
    unsigned  seq; /* For stable sorting only. */

    HB_INTERNAL static int cmp (const void *pa, const void *pb)
    {
      const feature_info_t *a = (const feature_info_t *) pa;
      const feature_info_t *b = (const feature_info_t *) pb;
      if (a->type != b->type) return (a->type < b->type ? -1 : 1);
      if (!a->is_exclusive &&
	  (a->setting & ~1) != (b->setting & ~1)) return (a->setting < b->setting ? -1 : 1);
	    return (a->seq < b->seq ? -1 : a->seq > b->seq ? 1 : 0);
    }

    /* compares type & setting only */
    int cmp (const feature_info_t& f) const
    {
      return (f.type != type) ? (f.type < type ? -1 : 1) :
	     (f.setting != setting) ? (f.setting < setting ? -1 : 1) : 0;
    }
  };

  struct feature_range_t
  {
    feature_info_t info;
    unsigned start;
    unsigned end;
  };

  private:
  struct feature_event_t
  {
    unsigned int index;
    bool start;
    feature_info_t feature;

    HB_INTERNAL static int cmp (const void *pa, const void *pb) {
      const feature_event_t *a = (const feature_event_t *) pa;
      const feature_event_t *b = (const feature_event_t *) pb;
      return a->index < b->index ? -1 : a->index > b->index ? 1 :
	     a->start < b->start ? -1 : a->start > b->start ? 1 :
	     feature_info_t::cmp (&a->feature, &b->feature);
    }
  };

  public:
  hb_face_t *face;
  hb_segment_properties_t props;

  public:
  hb_sorted_vector_t<feature_range_t> features;
  hb_sorted_vector_t<feature_info_t> current_features;
  unsigned range_first = HB_FEATURE_GLOBAL_START;
  unsigned range_last = HB_FEATURE_GLOBAL_END;
};


#endif /* HB_AAT_MAP_HH */
