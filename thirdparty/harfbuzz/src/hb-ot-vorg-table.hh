/*
 * Copyright © 2018 Adobe Inc.
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
 * Adobe Author(s): Michiharu Ariza
 */

#ifndef HB_OT_VORG_TABLE_HH
#define HB_OT_VORG_TABLE_HH

#include "hb-open-type.hh"

/*
 * VORG -- Vertical Origin Table
 * https://docs.microsoft.com/en-us/typography/opentype/spec/vorg
 */
#define HB_OT_TAG_VORG HB_TAG('V','O','R','G')

namespace OT {

struct VertOriginMetric
{
  int cmp (hb_codepoint_t g) const { return glyph.cmp (g); }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  public:
  HBGlyphID16	glyph;
  FWORD		vertOriginY;

  public:
  DEFINE_SIZE_STATIC (4);
};

struct VORG
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_VORG;

  bool has_data () const { return version.to_int (); }

  int get_y_origin (hb_codepoint_t glyph) const
  {
    unsigned int i;
    if (!vertYOrigins.bfind (glyph, &i))
      return defaultVertOriginY;
    return vertYOrigins[i].vertOriginY;
  }

  template <typename Iterator,
	    hb_requires (hb_is_iterator (Iterator))>
  void serialize (hb_serialize_context_t *c,
		  Iterator it,
		  FWORD defaultVertOriginY)
  {

    if (unlikely (!c->extend_min ((*this))))  return;

    this->version.major = 1;
    this->version.minor = 0;

    this->defaultVertOriginY = defaultVertOriginY;
    this->vertYOrigins.len = it.len ();

    c->copy_all (it);
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto *vorg_prime = c->serializer->start_embed<VORG> ();
    if (unlikely (!c->serializer->check_success (vorg_prime))) return_trace (false);

    auto it =
    + vertYOrigins.as_array ()
    | hb_filter (c->plan->glyphset (), &VertOriginMetric::glyph)
    | hb_map ([&] (const VertOriginMetric& _)
	      {
		hb_codepoint_t new_glyph = HB_SET_VALUE_INVALID;
		c->plan->new_gid_for_old_gid (_.glyph, &new_glyph);

		VertOriginMetric metric;
		metric.glyph = new_glyph;
		metric.vertOriginY = _.vertOriginY;
		return metric;
	      })
    ;

    /* serialize the new table */
    vorg_prime->serialize (c->serializer, it, defaultVertOriginY);
    return_trace (true);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  hb_barrier () &&
		  version.major == 1 &&
		  vertYOrigins.sanitize (c));
  }

  protected:
  FixedVersion<>version;	/* Version of VORG table. Set to 0x00010000u. */
  FWORD		defaultVertOriginY;
				/* The default vertical origin. */
  SortedArray16Of<VertOriginMetric>
		vertYOrigins;	/* The array of vertical origins. */

  public:
  DEFINE_SIZE_ARRAY(8, vertYOrigins);
};
} /* namespace OT */

#endif /* HB_OT_VORG_TABLE_HH */
