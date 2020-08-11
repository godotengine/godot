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
		  unsigned num_advances)
  {
    unsigned idx = 0;
    for (auto _ : it)
    {
      if (idx < num_advances)
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

    accelerator_t _mtx;
    _mtx.init (c->plan->source);
    unsigned num_advances = _mtx.num_advances_for_subset (c->plan);

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

    table_prime->serialize (c->serializer, it, num_advances);

    _mtx.fini ();

    if (unlikely (c->serializer->ran_out_of_room || c->serializer->in_error ()))
      return_trace (false);

    // Amend header num hmetrics
    if (unlikely (!subset_update_header (c->plan, num_advances)))
      return_trace (false);

    return_trace (true);
  }

  struct accelerator_t
  {
    friend struct hmtxvmtx;

    void init (hb_face_t *face,
	       unsigned int default_advance_ = 0)
    {
      default_advance = default_advance_ ? default_advance_ : hb_face_get_upem (face);

      num_advances = T::is_horizontal ? face->table.hhea->numberOfLongMetrics : face->table.vhea->numberOfLongMetrics;

      table = hb_sanitize_context_t ().reference_table<hmtxvmtx> (face, T::tableTag);

      /* Cap num_metrics() and num_advances() based on table length. */
      unsigned int len = table.get_length ();
      if (unlikely (num_advances * 4 > len))
	num_advances = len / 4;
      num_metrics = num_advances + (len - 4 * num_advances) / 2;

      /* We MUST set num_metrics to zero if num_advances is zero.
       * Our get_advance() depends on that. */
      if (unlikely (!num_advances))
      {
	num_metrics = num_advances = 0;
	table.destroy ();
	table = hb_blob_get_empty ();
      }

      var_table = hb_sanitize_context_t ().reference_table<HVARVVAR> (face, T::variationsTag);
    }

    void fini ()
    {
      table.destroy ();
      var_table.destroy ();
    }

    int get_side_bearing (hb_codepoint_t glyph) const
    {
      if (glyph < num_advances)
	return table->longMetricZ[glyph].sb;

      if (unlikely (glyph >= num_metrics))
	return 0;

      const FWORD *bearings = (const FWORD *) &table->longMetricZ[num_advances];
      return bearings[glyph - num_advances];
    }

    int get_side_bearing (hb_font_t *font, hb_codepoint_t glyph) const
    {
      int side_bearing = get_side_bearing (glyph);

#ifndef HB_NO_VAR
      if (unlikely (glyph >= num_metrics) || !font->num_coords)
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
      if (unlikely (glyph >= num_metrics))
      {
	/* If num_metrics is zero, it means we don't have the metrics table
	 * for this direction: return default advance.  Otherwise, it means that the
	 * glyph index is out of bound: return zero. */
	if (num_metrics)
	  return 0;
	else
	  return default_advance;
      }

      return table->longMetricZ[hb_min (glyph, (uint32_t) num_advances - 1)].advance;
    }

    unsigned int get_advance (hb_codepoint_t  glyph,
			      hb_font_t      *font) const
    {
      unsigned int advance = get_advance (glyph);

#ifndef HB_NO_VAR
      if (unlikely (glyph >= num_metrics) || !font->num_coords)
	return advance;

      if (var_table.get_length ())
	return advance + roundf (var_table->get_advance_var (glyph, font)); // TODO Optimize?!

      return _glyf_get_advance_var (font, glyph, T::tableTag == HB_OT_TAG_vmtx);
#else
      return advance;
#endif
    }

    unsigned int num_advances_for_subset (const hb_subset_plan_t *plan) const
    {
      unsigned int num_advances = plan->num_output_glyphs ();
      unsigned int last_advance = _advance_for_new_gid (plan,
							num_advances - 1);
      while (num_advances > 1 &&
	     last_advance == _advance_for_new_gid (plan,
						   num_advances - 2))
      {
	num_advances--;
      }

      return num_advances;
    }

    private:
    unsigned int _advance_for_new_gid (const hb_subset_plan_t *plan,
				       hb_codepoint_t new_gid) const
    {
      hb_codepoint_t old_gid;
      if (!plan->old_gid_for_new_gid (new_gid, &old_gid))
	return 0;

      return get_advance (old_gid);
    }

    protected:
    unsigned int num_metrics;
    unsigned int num_advances;
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

struct hmtx_accelerator_t : hmtx::accelerator_t {};
struct vmtx_accelerator_t : vmtx::accelerator_t {};

} /* namespace OT */


#endif /* HB_OT_HMTX_TABLE_HH */
