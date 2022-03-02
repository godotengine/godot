/*
 * Copyright © 2011,2012  Google, Inc.
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
 * Google Author(s): Behdad Esfahbod, Roderick Sheeter
 */

#ifndef HB_OT_HMTX_TABLE_HH
#define HB_OT_HMTX_TABLE_HH

#include "hb-open-type.hh"
#include "hb-ot-maxp-table.hh"
#include "hb-ot-hhea-table.hh"
#include "hb-ot-var-hvar-table.hh"
#include "hb-ot-metrics.hh"

/*
 * hmtx -- Horizontal Metrics
 * https://docs.microsoft.com/en-us/typography/opentype/spec/hmtx
 * vmtx -- Vertical Metrics
 * https://docs.microsoft.com/en-us/typography/opentype/spec/vmtx
 */
#define HB_OT_TAG_hmtx HB_TAG('h','m','t','x')
#define HB_OT_TAG_vmtx HB_TAG('v','m','t','x')


HB_INTERNAL int
_glyf_get_side_bearing_var (hb_font_t *font, hb_codepoint_t glyph, bool is_vertical);

HB_INTERNAL unsigned
_glyf_get_advance_var (hb_font_t *font, hb_codepoint_t glyph, bool is_vertical);


namespace OT {


struct LongMetric
{
  UFWORD	advance; /* Advance width/height. */
  FWORD		sb; /* Leading (left/top) side bearing. */
  public:
  DEFINE_SIZE_STATIC (4);
};


template <typename T, typename H>
struct hmtxvmtx
{
  bool sanitize (hb_sanitize_context_t *c HB_UNUSED) const
  {
    TRACE_SANITIZE (this);
    /* We don't check for anything specific here.  The users of the
     * struct do all the hard work... */
    return_trace (true);
  }


  bool subset_update_header (hb_subset_plan_t *plan,
			     unsigned int num_hmetrics) const
  {
    hb_blob_t *src_blob = hb_sanitize_context_t ().reference_table<H> (plan->source, H::tableTag);
    hb_blob_t *dest_blob = hb_blob_copy_writable_or_fail (src_blob);
    hb_blob_destroy (src_blob);

    if (unlikely (!dest_blob)) {
      return false;
    }

    unsigned int length;
    H *table = (H *) hb_blob_get_data (dest_blob, &length);
    table->numberOfLongMetrics = num_hmetrics;

    bool result = plan->add_table (H::tableTag, dest_blob);
    hb_blob_destroy (dest_blob);

    return result;
  }

  template<typename Iterator,
	   hb_requires (hb_is_iterator (Iterator))>
  void serialize (hb_serialize_context_t *c,
		  Iterator it,
		  unsigned num_long_metrics)
  {
    unsigned idx = 0;
    for (auto _ : it)
    {
      if (idx < num_long_metrics)
      {
	LongMetric lm;
	lm.advance = _.first;
	lm.sb = _.second;
	if (unlikely (!c->embed<LongMetric> (&lm))) return;
      }
      else
      {
	FWORD *sb = c->allocate_size<FWORD> (FWORD::static_size);
	if (unlikely (!sb)) return;
	*sb = _.second;
      }
      idx++;
    }
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);

    T *table_prime = c->serializer->start_embed <T> ();
    if (unlikely (!table_prime)) return_trace (false);

    accelerator_t _mtx (c->plan->source);
    unsigned num_long_metrics;
    {
      /* Determine num_long_metrics to encode. */
      auto& plan = c->plan;
      num_long_metrics = plan->num_output_glyphs ();
      hb_codepoint_t old_gid = 0;
      unsigned int last_advance = plan->old_gid_for_new_gid (num_long_metrics - 1, &old_gid) ? _mtx.get_advance (old_gid) : 0;
      while (num_long_metrics > 1 &&
	     last_advance == (plan->old_gid_for_new_gid (num_long_metrics - 2, &old_gid) ? _mtx.get_advance (old_gid) : 0))
      {
	num_long_metrics--;
      }
    }

    auto it =
    + hb_range (c->plan->num_output_glyphs ())
    | hb_map ([c, &_mtx] (unsigned _)
	      {
		hb_codepoint_t old_gid;
		if (!c->plan->old_gid_for_new_gid (_, &old_gid))
		  return hb_pair (0u, 0);
		return hb_pair (_mtx.get_advance (old_gid), _mtx.get_side_bearing (old_gid));
	      })
    ;

    table_prime->serialize (c->serializer, it, num_long_metrics);

    if (unlikely (c->serializer->in_error ()))
      return_trace (false);

    // Amend header num hmetrics
    if (unlikely (!subset_update_header (c->plan, num_long_metrics)))
      return_trace (false);

    return_trace (true);
  }

  struct accelerator_t
  {
    friend struct hmtxvmtx;

    accelerator_t (hb_face_t *face,
		   unsigned int default_advance_ = 0)
    {
      table = hb_sanitize_context_t ().reference_table<hmtxvmtx> (face, T::tableTag);
      var_table = hb_sanitize_context_t ().reference_table<HVARVVAR> (face, T::variationsTag);

      default_advance = default_advance_ ? default_advance_ : hb_face_get_upem (face);

      /* Populate count variables and sort them out as we go */

      unsigned int len = table.get_length ();
      if (len & 1)
        len--;

      num_long_metrics = T::is_horizontal ?
			 face->table.hhea->numberOfLongMetrics :
#ifndef HB_NO_VERTICAL
			 face->table.vhea->numberOfLongMetrics
#else
			 0
#endif
			 ;
      if (unlikely (num_long_metrics * 4 > len))
	num_long_metrics = len / 4;
      len -= num_long_metrics * 4;

      num_bearings = face->table.maxp->get_num_glyphs ();

      if (unlikely (num_bearings < num_long_metrics))
        num_bearings = num_long_metrics;
      if (unlikely ((num_bearings - num_long_metrics) * 2 > len))
        num_bearings = num_long_metrics + len / 2;
      len -= (num_bearings - num_long_metrics) * 2;

      /* We MUST set num_bearings to zero if num_long_metrics is zero.
       * Our get_advance() depends on that. */
      if (unlikely (!num_long_metrics))
	num_bearings = num_long_metrics = 0;

      num_advances = num_bearings + len / 2;
      num_glyphs = face->get_num_glyphs ();
      if (num_glyphs < num_advances)
        num_glyphs = num_advances;
    }
    ~accelerator_t ()
    {
      table.destroy ();
      var_table.destroy ();
    }

    int get_side_bearing (hb_codepoint_t glyph) const
    {
      if (glyph < num_long_metrics)
	return table->longMetricZ[glyph].sb;

      if (unlikely (glyph >= num_bearings))
	return 0;

      const FWORD *bearings = (const FWORD *) &table->longMetricZ[num_long_metrics];
      return bearings[glyph - num_long_metrics];
    }

    int get_side_bearing (hb_font_t *font, hb_codepoint_t glyph) const
    {
      int side_bearing = get_side_bearing (glyph);

#ifndef HB_NO_VAR
      if (unlikely (glyph >= num_bearings) || !font->num_coords)
	return side_bearing;

      if (var_table.get_length ())
	return side_bearing + var_table->get_side_bearing_var (glyph, font->coords, font->num_coords); // TODO Optimize?!

      return _glyf_get_side_bearing_var (font, glyph, T::tableTag == HB_OT_TAG_vmtx);
#else
      return side_bearing;
#endif
    }

    unsigned int get_advance (hb_codepoint_t glyph) const
    {
      /* OpenType case. */
      if (glyph < num_bearings)
	return table->longMetricZ[hb_min (glyph, (uint32_t) num_long_metrics - 1)].advance;

      /* If num_advances is zero, it means we don't have the metrics table
       * for this direction: return default advance.  Otherwise, there's a
       * well-defined answer. */
      if (unlikely (!num_advances))
	return default_advance;

#ifdef HB_NO_BORING_EXPANSION
      return 0;
#endif

      if (unlikely (glyph >= num_glyphs))
        return 0;

      /* num_bearings <= glyph < num_glyphs;
       * num_bearings <= num_advances */

      /* TODO Optimize */

      if (num_bearings == num_advances)
        return get_advance (num_bearings - 1);

      const FWORD *bearings = (const FWORD *) &table->longMetricZ[num_long_metrics];
      const UFWORD *advances = (const UFWORD *) &bearings[num_bearings - num_long_metrics];

      return advances[hb_min (glyph - num_bearings, num_advances - num_bearings - 1)];
    }

    unsigned int get_advance (hb_codepoint_t  glyph,
			      hb_font_t      *font) const
    {
      unsigned int advance = get_advance (glyph);

#ifndef HB_NO_VAR
      if (unlikely (glyph >= num_bearings) || !font->num_coords)
	return advance;

      if (var_table.get_length ())
	return advance + roundf (var_table->get_advance_var (glyph, font)); // TODO Optimize?!

      return _glyf_get_advance_var (font, glyph, T::tableTag == HB_OT_TAG_vmtx);
#else
      return advance;
#endif
    }

    protected:
    // 0 <= num_long_metrics <= num_bearings <= num_advances <= num_glyphs
    unsigned num_long_metrics;
    unsigned num_bearings;
    unsigned num_advances;
    unsigned num_glyphs;

    unsigned int default_advance;

    private:
    hb_blob_ptr_t<hmtxvmtx> table;
    hb_blob_ptr_t<HVARVVAR> var_table;
  };

  protected:
  UnsizedArrayOf<LongMetric>
		longMetricZ;	/* Paired advance width and leading
				 * bearing values for each glyph. The
				 * value numOfHMetrics comes from
				 * the 'hhea' table. If the font is
				 * monospaced, only one entry need
				 * be in the array, but that entry is
				 * required. The last entry applies to
				 * all subsequent glyphs. */
/*UnsizedArrayOf<FWORD>	leadingBearingX;*/
				/* Here the advance is assumed
				 * to be the same as the advance
				 * for the last entry above. The
				 * number of entries in this array is
				 * derived from numGlyphs (from 'maxp'
				 * table) minus numberOfLongMetrics.
				 * This generally is used with a run
				 * of monospaced glyphs (e.g., Kanji
				 * fonts or Courier fonts). Only one
				 * run is allowed and it must be at
				 * the end. This allows a monospaced
				 * font to vary the side bearing
				 * values for each glyph. */
/*UnsizedArrayOf<UFWORD>advancesX;*/
				/* TODO Document. */
  public:
  DEFINE_SIZE_ARRAY (0, longMetricZ);
};

struct hmtx : hmtxvmtx<hmtx, hhea> {
  static constexpr hb_tag_t tableTag = HB_OT_TAG_hmtx;
  static constexpr hb_tag_t variationsTag = HB_OT_TAG_HVAR;
  static constexpr bool is_horizontal = true;
};
struct vmtx : hmtxvmtx<vmtx, vhea> {
  static constexpr hb_tag_t tableTag = HB_OT_TAG_vmtx;
  static constexpr hb_tag_t variationsTag = HB_OT_TAG_VVAR;
  static constexpr bool is_horizontal = false;
};

struct hmtx_accelerator_t : hmtx::accelerator_t {
  hmtx_accelerator_t (hb_face_t *face) : hmtx::accelerator_t (face) {}
};
struct vmtx_accelerator_t : vmtx::accelerator_t {
  vmtx_accelerator_t (hb_face_t *face) : vmtx::accelerator_t (face) {}
};

} /* namespace OT */


#endif /* HB_OT_HMTX_TABLE_HH */
