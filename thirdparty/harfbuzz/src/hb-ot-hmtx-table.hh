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

#include "hb-open-type-private.hh"
#include "hb-ot-hhea-table.hh"
#include "hb-ot-os2-table.hh"
#include "hb-ot-var-hvar-table.hh"
#include "hb-subset-plan.hh"

/*
 * hmtx -- Horizontal Metrics
 * https://docs.microsoft.com/en-us/typography/opentype/spec/hmtx
 * vmtx -- Vertical Metrics
 * https://docs.microsoft.com/en-us/typography/opentype/spec/vmtx
 */
#define HB_OT_TAG_hmtx HB_TAG('h','m','t','x')
#define HB_OT_TAG_vmtx HB_TAG('v','m','t','x')


namespace OT {


struct LongMetric
{
  UFWORD	advance; /* Advance width/height. */
  FWORD		lsb; /* Leading (left/top) side bearing. */
  public:
  DEFINE_SIZE_STATIC (4);
};

template <typename T, typename H>
struct hmtxvmtx
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    /* We don't check for anything specific here.  The users of the
     * struct do all the hard work... */
    return_trace (true);
  }


  inline bool subset_update_header (hb_subset_plan_t *plan,
                                    unsigned int num_hmetrics) const
  {
    hb_blob_t *src_blob = hb_sanitize_context_t().reference_table<H> (plan->source, H::tableTag);
    hb_blob_t *dest_blob = hb_blob_copy_writable_or_fail(src_blob);
    hb_blob_destroy (src_blob);

    if (unlikely (!dest_blob)) {
      return false;
    }

    unsigned int length;
    H *table = (H *) hb_blob_get_data (dest_blob, &length);
    table->numberOfLongMetrics.set (num_hmetrics);

    bool result = plan->add_table (H::tableTag, dest_blob);
    hb_blob_destroy (dest_blob);

    return result;
  }

  inline bool subset (hb_subset_plan_t *plan) const
  {
    typename T::accelerator_t _mtx;
    _mtx.init (plan->source);

    /* All the trailing glyphs with the same advance can use one LongMetric
     * and just keep LSB */
    hb_vector_t<hb_codepoint_t> &gids = plan->glyphs;
    unsigned int num_advances = gids.len;
    unsigned int last_advance = _mtx.get_advance (gids[num_advances - 1]);
    while (num_advances > 1
        && last_advance == _mtx.get_advance (gids[num_advances - 2]))
    {
      num_advances--;
    }

    /* alloc the new table */
    size_t dest_sz = num_advances * 4
                  + (gids.len - num_advances) * 2;
    void *dest = (void *) malloc (dest_sz);
    if (unlikely (!dest))
    {
      return false;
    }
    DEBUG_MSG(SUBSET, nullptr, "%c%c%c%c in src has %d advances, %d lsbs", HB_UNTAG(T::tableTag), _mtx.num_advances, _mtx.num_metrics - _mtx.num_advances);
    DEBUG_MSG(SUBSET, nullptr, "%c%c%c%c in dest has %d advances, %d lsbs, %u bytes", HB_UNTAG(T::tableTag), num_advances, gids.len - num_advances, (unsigned int) dest_sz);

    const char *source_table = hb_blob_get_data (_mtx.blob, nullptr);
    // Copy everything over
    LongMetric * old_metrics = (LongMetric *) source_table;
    FWORD *lsbs = (FWORD *) (old_metrics + _mtx.num_advances);
    char * dest_pos = (char *) dest;

    bool failed = false;
    for (unsigned int i = 0; i < gids.len; i++)
    {
      /* the last metric or the one for gids[i] */
      LongMetric *src_metric = old_metrics + MIN ((hb_codepoint_t) _mtx.num_advances - 1, gids[i]);
      if (gids[i] < _mtx.num_advances)
      {
        /* src is a LongMetric */
        if (i < num_advances)
        {
          /* dest is a LongMetric, copy it */
          *((LongMetric *) dest_pos) = *src_metric;
        }
        else
        {
          /* dest just lsb */
          *((FWORD *) dest_pos) = src_metric->lsb;
        }
      }
      else
      {
	if (gids[i] >= _mtx.num_metrics)
	{
	  DEBUG_MSG(SUBSET, nullptr, "gid %d is >= number of source metrics %d",
		    gids[i], _mtx.num_metrics);
	  failed = true;
	  break;
	}
	FWORD src_lsb = *(lsbs + gids[i] - _mtx.num_advances);
        if (i < num_advances)
        {
          /* dest needs a full LongMetric */
          LongMetric *metric = (LongMetric *)dest_pos;
          metric->advance = src_metric->advance;
          metric->lsb = src_lsb;
        }
        else
        {
          /* dest just needs an lsb */
          *((FWORD *) dest_pos) = src_lsb;
        }
      }
      dest_pos += (i < num_advances ? 4 : 2);
    }
    _mtx.fini ();

    // Amend header num hmetrics
    if (failed || unlikely (!subset_update_header (plan, num_advances)))
    {
      free (dest);
      return false;
    }

    hb_blob_t *result = hb_blob_create ((const char *)dest,
                                        dest_sz,
                                        HB_MEMORY_MODE_READONLY,
                                        dest,
                                        free);
    bool success = plan->add_table (T::tableTag, result);
    hb_blob_destroy (result);
    return success;
  }

  struct accelerator_t
  {
    friend struct hmtxvmtx;

    inline void init (hb_face_t *face,
		      unsigned int default_advance_ = 0)
    {
      default_advance = default_advance_ ? default_advance_ : hb_face_get_upem (face);

      bool got_font_extents = false;
      if (T::os2Tag)
      {
	hb_blob_t *os2_blob = hb_sanitize_context_t().reference_table<os2> (face);
	const os2 *os2_table = os2_blob->as<os2> ();
#define USE_TYPO_METRICS (1u<<7)
	if (0 != (os2_table->fsSelection & USE_TYPO_METRICS))
	{
	  ascender = os2_table->sTypoAscender;
	  descender = os2_table->sTypoDescender;
	  line_gap = os2_table->sTypoLineGap;
	  got_font_extents = (ascender | descender) != 0;
	}
	hb_blob_destroy (os2_blob);
      }

      hb_blob_t *_hea_blob = hb_sanitize_context_t().reference_table<H> (face);
      const H *_hea_table = _hea_blob->as<H> ();
      num_advances = _hea_table->numberOfLongMetrics;
      if (!got_font_extents)
      {
	ascender = _hea_table->ascender;
	descender = _hea_table->descender;
	line_gap = _hea_table->lineGap;
	got_font_extents = (ascender | descender) != 0;
      }
      hb_blob_destroy (_hea_blob);

      has_font_extents = got_font_extents;

      blob = hb_sanitize_context_t().reference_table<hmtxvmtx> (face, T::tableTag);

      /* Cap num_metrics() and num_advances() based on table length. */
      unsigned int len = hb_blob_get_length (blob);
      if (unlikely (num_advances * 4 > len))
	num_advances = len / 4;
      num_metrics = num_advances + (len - 4 * num_advances) / 2;

      /* We MUST set num_metrics to zero if num_advances is zero.
       * Our get_advance() depends on that. */
      if (unlikely (!num_advances))
      {
	num_metrics = num_advances = 0;
	hb_blob_destroy (blob);
	blob = hb_blob_get_empty ();
      }
      table = blob->as<hmtxvmtx> ();

      var_blob = hb_sanitize_context_t().reference_table<HVARVVAR> (face, T::variationsTag);
      var_table = var_blob->as<HVARVVAR> ();
    }

    inline void fini (void)
    {
      hb_blob_destroy (blob);
      hb_blob_destroy (var_blob);
    }

    inline unsigned int get_advance (hb_codepoint_t  glyph) const
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

      return table->longMetric[MIN (glyph, (uint32_t) num_advances - 1)].advance;
    }

    inline unsigned int get_advance (hb_codepoint_t  glyph,
                                     hb_font_t      *font) const
    {
      unsigned int advance = get_advance (glyph);
      if (likely(glyph < num_metrics))
      {
        advance += (font->num_coords ? var_table->get_advance_var (glyph, font->coords, font->num_coords) : 0); // TODO Optimize?!
      }
      return advance;
    }

    public:
    bool has_font_extents;
    unsigned short ascender;
    unsigned short descender;
    unsigned short line_gap;

    protected:
    unsigned int num_metrics;
    unsigned int num_advances;
    unsigned int default_advance;

    private:
    const hmtxvmtx *table;
    hb_blob_t *blob;
    const HVARVVAR *var_table;
    hb_blob_t *var_blob;
  };

  protected:
  LongMetric	longMetric[VAR];	/* Paired advance width and leading
					 * bearing values for each glyph. The
					 * value numOfHMetrics comes from
					 * the 'hhea' table. If the font is
					 * monospaced, only one entry need
					 * be in the array, but that entry is
					 * required. The last entry applies to
					 * all subsequent glyphs. */
/*FWORD		leadingBearingX[VAR];*/	/* Here the advance is assumed
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
  DEFINE_SIZE_ARRAY (0, longMetric);
};

struct hmtx : hmtxvmtx<hmtx, hhea> {
  static const hb_tag_t tableTag	= HB_OT_TAG_hmtx;
  static const hb_tag_t variationsTag	= HB_OT_TAG_HVAR;
  static const hb_tag_t os2Tag		= HB_OT_TAG_os2;
};
struct vmtx : hmtxvmtx<vmtx, vhea> {
  static const hb_tag_t tableTag	= HB_OT_TAG_vmtx;
  static const hb_tag_t variationsTag	= HB_OT_TAG_VVAR;
  static const hb_tag_t os2Tag		= HB_TAG_NONE;
};

} /* namespace OT */


#endif /* HB_OT_HMTX_TABLE_HH */
