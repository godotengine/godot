/*
 * Copyright © 2017  Google, Inc.
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

#ifndef HB_OT_VAR_HVAR_TABLE_HH
#define HB_OT_VAR_HVAR_TABLE_HH

#include "hb-ot-layout-common-private.hh"


namespace OT {


struct DeltaSetIndexMap
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  c->check_array (mapData, get_width (), mapCount));
  }

  unsigned int map (unsigned int v) const /* Returns 16.16 outer.inner. */
  {
    /* If count is zero, pass value unchanged.  This takes
     * care of direct mapping for advance map. */
    if (!mapCount)
      return v;

    if (v >= mapCount)
      v = mapCount - 1;

    unsigned int u = 0;
    { /* Fetch it. */
      unsigned int w = get_width ();
      const HBUINT8 *p = mapData + w * v;
      for (; w; w--)
	u = (u << 8) + *p++;
    }

    { /* Repack it. */
      unsigned int n = get_inner_bitcount ();
      unsigned int outer = u >> n;
      unsigned int inner = u & ((1 << n) - 1);
      u = (outer<<16) | inner;
    }

    return u;
  }

  protected:
  inline unsigned int get_width (void) const
  { return ((format >> 4) & 3) + 1; }

  inline unsigned int get_inner_bitcount (void) const
  { return (format & 0xF) + 1; }

  protected:
  HBUINT16	format;		/* A packed field that describes the compressed
				 * representation of delta-set indices. */
  HBUINT16	mapCount;	/* The number of mapping entries. */
  HBUINT8		mapData[VAR];	/* The delta-set index mapping data. */

  public:
  DEFINE_SIZE_ARRAY (4, mapData);
};


/*
 * HVAR -- Horizontal Metrics Variations
 * https://docs.microsoft.com/en-us/typography/opentype/spec/hvar
 * VVAR -- Vertical Metrics Variations
 * https://docs.microsoft.com/en-us/typography/opentype/spec/vvar
 */
#define HB_OT_TAG_HVAR HB_TAG('H','V','A','R')
#define HB_OT_TAG_VVAR HB_TAG('V','V','A','R')

struct HVARVVAR
{
  static const hb_tag_t HVARTag	= HB_OT_TAG_HVAR;
  static const hb_tag_t VVARTag	= HB_OT_TAG_VVAR;

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (version.sanitize (c) &&
		  likely (version.major == 1) &&
		  varStore.sanitize (c, this) &&
		  advMap.sanitize (c, this) &&
		  lsbMap.sanitize (c, this) &&
		  rsbMap.sanitize (c, this));
  }

  inline float get_advance_var (hb_codepoint_t glyph,
				int *coords, unsigned int coord_count) const
  {
    unsigned int varidx = (this+advMap).map (glyph);
    return (this+varStore).get_delta (varidx, coords, coord_count);
  }

  inline bool has_sidebearing_deltas (void) const
  { return lsbMap && rsbMap; }

  protected:
  FixedVersion<>version;	/* Version of the metrics variation table
				 * initially set to 0x00010000u */
  LOffsetTo<VariationStore>
		varStore;	/* Offset to item variation store table. */
  LOffsetTo<DeltaSetIndexMap>
		advMap;		/* Offset to advance var-idx mapping. */
  LOffsetTo<DeltaSetIndexMap>
		lsbMap;		/* Offset to lsb/tsb var-idx mapping. */
  LOffsetTo<DeltaSetIndexMap>
		rsbMap;		/* Offset to rsb/bsb var-idx mapping. */

  public:
  DEFINE_SIZE_STATIC (20);
};

struct HVAR : HVARVVAR {
  static const hb_tag_t tableTag	= HB_OT_TAG_HVAR;
};
struct VVAR : HVARVVAR {
  static const hb_tag_t tableTag	= HB_OT_TAG_VVAR;

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (static_cast<const HVARVVAR *> (this)->sanitize (c) &&
		  vorgMap.sanitize (c, this));
  }

  protected:
  LOffsetTo<DeltaSetIndexMap>
		vorgMap;	/* Offset to vertical-origin var-idx mapping. */

  public:
  DEFINE_SIZE_STATIC (24);
};

} /* namespace OT */


#endif /* HB_OT_VAR_HVAR_TABLE_HH */
