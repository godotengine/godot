/*
 * Copyright © 2011,2012  Google, Inc.
 * Copyright © 2018  Ebrahim Byagowi
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

#ifndef HB_OT_OS2_TABLE_HH
#define HB_OT_OS2_TABLE_HH

#include "hb-open-type.hh"
#include "hb-ot-os2-unicode-ranges.hh"
#include "hb-ot-var-mvar-table.hh"

#include "hb-set.hh"

/*
 * OS/2 and Windows Metrics
 * https://docs.microsoft.com/en-us/typography/opentype/spec/os2
 */
#define HB_OT_TAG_OS2 HB_TAG('O','S','/','2')


namespace OT {

struct OS2V1Tail
{
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  public:
  HBUINT32	ulCodePageRange1;
  HBUINT32	ulCodePageRange2;
  public:
  DEFINE_SIZE_STATIC (8);
};

struct OS2V2Tail
{
  bool has_data () const { return sxHeight || sCapHeight; }

  const OS2V2Tail * operator -> () const { return this; }
  OS2V2Tail * operator -> () { return this; }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  public:
  HBINT16	sxHeight;
  HBINT16	sCapHeight;
  HBUINT16	usDefaultChar;
  HBUINT16	usBreakChar;
  HBUINT16	usMaxContext;
  public:
  DEFINE_SIZE_STATIC (10);
};

struct OS2V5Tail
{
  inline bool get_optical_size (unsigned int *lower, unsigned int *upper) const
  {
    unsigned int lower_optical_size = usLowerOpticalPointSize;
    unsigned int upper_optical_size = usUpperOpticalPointSize;

    /* Per https://docs.microsoft.com/en-us/typography/opentype/spec/os2#lps */
    if (lower_optical_size < upper_optical_size &&
	lower_optical_size >= 1 && lower_optical_size <= 0xFFFE &&
	upper_optical_size >= 2 && upper_optical_size <= 0xFFFF)
    {
      *lower = lower_optical_size;
      *upper = upper_optical_size;
      return true;
    }
    return false;
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  public:
  HBUINT16	usLowerOpticalPointSize;
  HBUINT16	usUpperOpticalPointSize;
  public:
  DEFINE_SIZE_STATIC (4);
};

struct OS2
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_OS2;

  bool has_data () const { return usWeightClass || usWidthClass || usFirstCharIndex || usLastCharIndex; }

  const OS2V1Tail &v1 () const { return version >= 1 ? v1X : Null (OS2V1Tail); }
  const OS2V2Tail &v2 () const { return version >= 2 ? v2X : Null (OS2V2Tail); }
  const OS2V5Tail &v5 () const { return version >= 5 ? v5X : Null (OS2V5Tail); }

  enum selection_flag_t {
    ITALIC		= 1u<<0,
    UNDERSCORE		= 1u<<1,
    NEGATIVE		= 1u<<2,
    OUTLINED		= 1u<<3,
    STRIKEOUT		= 1u<<4,
    BOLD		= 1u<<5,
    REGULAR		= 1u<<6,
    USE_TYPO_METRICS	= 1u<<7,
    WWS			= 1u<<8,
    OBLIQUE		= 1u<<9
  };

  bool        is_italic () const { return fsSelection & ITALIC; }
  bool       is_oblique () const { return fsSelection & OBLIQUE; }
  bool use_typo_metrics () const { return fsSelection & USE_TYPO_METRICS; }

  enum width_class_t {
    FWIDTH_ULTRA_CONDENSED	= 1, /* 50% */
    FWIDTH_EXTRA_CONDENSED	= 2, /* 62.5% */
    FWIDTH_CONDENSED		= 3, /* 75% */
    FWIDTH_SEMI_CONDENSED	= 4, /* 87.5% */
    FWIDTH_NORMAL		= 5, /* 100% */
    FWIDTH_SEMI_EXPANDED	= 6, /* 112.5% */
    FWIDTH_EXPANDED		= 7, /* 125% */
    FWIDTH_EXTRA_EXPANDED	= 8, /* 150% */
    FWIDTH_ULTRA_EXPANDED	= 9  /* 200% */
  };

  float get_width () const
  {
    switch (usWidthClass) {
    case FWIDTH_ULTRA_CONDENSED:return 50.f;
    case FWIDTH_EXTRA_CONDENSED:return 62.5f;
    case FWIDTH_CONDENSED:	return 75.f;
    case FWIDTH_SEMI_CONDENSED:	return 87.5f;
    default:
    case FWIDTH_NORMAL:		return 100.f;
    case FWIDTH_SEMI_EXPANDED:	return 112.5f;
    case FWIDTH_EXPANDED:	return 125.f;
    case FWIDTH_EXTRA_EXPANDED:	return 150.f;
    case FWIDTH_ULTRA_EXPANDED:	return 200.f;
    }
  }

  float map_wdth_to_widthclass(float width) const
  {
    if (width < 50) return 1.0f;
    if (width > 200) return 9.0f;

    float ratio = (width - 50) / 12.5f;
    int a = (int) floorf (ratio);
    int b = (int) ceilf (ratio);

    /* follow this maping:
     * https://docs.microsoft.com/en-us/typography/opentype/spec/os2#uswidthclass
     */
    if (b <= 6) // 50-125
    {
      if (a == b) return a + 1.0f;
    }
    else if (b == 7) // no mapping for 137.5
    {
      a = 6;
      b = 8;
    }
    else if (b == 8)
    {
      if (a == b) return 8.0f; // 150
      a = 6;
    }
    else
    {
      if (a == b && a == 12) return 9.0f; //200
      b = 12;
      a = 8;
    }

    float va = 50 + a * 12.5f;
    float vb = 50 + b * 12.5f;

    float ret =  a + (width - va) / (vb - va);
    if (a <= 6) ret += 1.0f;
    return ret;
  }

  static unsigned calc_avg_char_width (const hb_hashmap_t<hb_codepoint_t, hb_pair_t<unsigned, int>>& hmtx_map)
  {
    unsigned num = 0;
    unsigned total_width = 0;
    for (const auto& _ : hmtx_map.values_ref ())
    {
      unsigned width = _.first;
      if (width)
      {
        total_width += width;
        num++;
      }
    }

    return num ? (unsigned) roundf ((double) total_width / (double) num) : 0;
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    OS2 *os2_prime = c->serializer->embed (this);
    if (unlikely (!os2_prime)) return_trace (false);

#ifndef HB_NO_VAR
    if (c->plan->normalized_coords)
    {
      auto &MVAR = *c->plan->source->table.MVAR;
      auto *table = os2_prime;

      HB_ADD_MVAR_VAR (HB_OT_METRICS_TAG_HORIZONTAL_ASCENDER,         sTypoAscender);
      HB_ADD_MVAR_VAR (HB_OT_METRICS_TAG_HORIZONTAL_DESCENDER,        sTypoDescender);
      HB_ADD_MVAR_VAR (HB_OT_METRICS_TAG_HORIZONTAL_LINE_GAP,         sTypoLineGap);
      HB_ADD_MVAR_VAR (HB_OT_METRICS_TAG_HORIZONTAL_CLIPPING_ASCENT,  usWinAscent);
      HB_ADD_MVAR_VAR (HB_OT_METRICS_TAG_HORIZONTAL_CLIPPING_DESCENT, usWinDescent);
      HB_ADD_MVAR_VAR (HB_OT_METRICS_TAG_SUBSCRIPT_EM_X_SIZE,         ySubscriptXSize);
      HB_ADD_MVAR_VAR (HB_OT_METRICS_TAG_SUBSCRIPT_EM_Y_SIZE,         ySubscriptYSize);
      HB_ADD_MVAR_VAR (HB_OT_METRICS_TAG_SUBSCRIPT_EM_X_OFFSET,       ySubscriptXOffset);
      HB_ADD_MVAR_VAR (HB_OT_METRICS_TAG_SUBSCRIPT_EM_Y_OFFSET,       ySubscriptYOffset);
      HB_ADD_MVAR_VAR (HB_OT_METRICS_TAG_SUPERSCRIPT_EM_X_SIZE,       ySuperscriptXSize);
      HB_ADD_MVAR_VAR (HB_OT_METRICS_TAG_SUPERSCRIPT_EM_Y_SIZE,       ySuperscriptYSize);
      HB_ADD_MVAR_VAR (HB_OT_METRICS_TAG_SUPERSCRIPT_EM_X_OFFSET,     ySuperscriptXOffset);
      HB_ADD_MVAR_VAR (HB_OT_METRICS_TAG_SUPERSCRIPT_EM_Y_OFFSET,     ySuperscriptYOffset);
      HB_ADD_MVAR_VAR (HB_OT_METRICS_TAG_STRIKEOUT_SIZE,              yStrikeoutSize);
      HB_ADD_MVAR_VAR (HB_OT_METRICS_TAG_STRIKEOUT_OFFSET,            yStrikeoutPosition);

      if (os2_prime->version >= 2)
      {
        hb_barrier ();
        auto *table = & const_cast<OS2V2Tail &> (os2_prime->v2 ());
        HB_ADD_MVAR_VAR (HB_OT_METRICS_TAG_X_HEIGHT,                   sxHeight);
        HB_ADD_MVAR_VAR (HB_OT_METRICS_TAG_CAP_HEIGHT,                 sCapHeight);
      }

      unsigned avg_char_width = calc_avg_char_width (c->plan->hmtx_map);
      if (!c->serializer->check_assign (os2_prime->xAvgCharWidth, avg_char_width,
                                        HB_SERIALIZE_ERROR_INT_OVERFLOW))
        return_trace (false);
    }
#endif

    Triple *axis_range;
    if (c->plan->user_axes_location.has (HB_TAG ('w','g','h','t'), &axis_range))
    {
      unsigned weight_class = static_cast<unsigned> (roundf (hb_clamp (axis_range->middle, 1.0, 1000.0)));
      if (os2_prime->usWeightClass != weight_class)
        os2_prime->usWeightClass = weight_class;
    }

    if (c->plan->user_axes_location.has (HB_TAG ('w','d','t','h'), &axis_range))
    {
      unsigned width_class = static_cast<unsigned> (roundf (map_wdth_to_widthclass (axis_range->middle)));
      if (os2_prime->usWidthClass != width_class)
        os2_prime->usWidthClass = width_class;
    }

    os2_prime->usFirstCharIndex = hb_min (0xFFFFu, c->plan->unicodes.get_min ());
    os2_prime->usLastCharIndex  = hb_min (0xFFFFu, c->plan->unicodes.get_max ());

    if (c->plan->flags & HB_SUBSET_FLAGS_NO_PRUNE_UNICODE_RANGES)
      return_trace (true);

    _update_unicode_ranges (&c->plan->unicodes, os2_prime->ulUnicodeRange);

    return_trace (true);
  }

  void _update_unicode_ranges (const hb_set_t *codepoints,
			       HBUINT32 ulUnicodeRange[4]) const
  {
    HBUINT32 newBits[4];
    for (unsigned int i = 0; i < 4; i++)
      newBits[i] = 0;

    /* This block doesn't show up in profiles. If it ever did,
     * we can rewrite it to iterate over OS/2 ranges and use
     * set iteration to check if the range matches. */
    for (auto cp : *codepoints)
    {
      unsigned int bit = _hb_ot_os2_get_unicode_range_bit (cp);
      if (bit < 128)
      {
	unsigned int block = bit / 32;
	unsigned int bit_in_block = bit % 32;
	unsigned int mask = 1 << bit_in_block;
	newBits[block] = newBits[block] | mask;
      }
      if (cp >= 0x10000 && cp <= 0x110000)
      {
	/* the spec says that bit 57 ("Non Plane 0") implies that there's
	   at least one codepoint beyond the BMP; so I also include all
	   the non-BMP codepoints here */
	newBits[1] = newBits[1] | (1 << 25);
      }
    }

    for (unsigned int i = 0; i < 4; i++)
      ulUnicodeRange[i] = ulUnicodeRange[i] & newBits[i]; // set bits only if set in the original
  }

  /* https://github.com/Microsoft/Font-Validator/blob/520aaae/OTFontFileVal/val_OS2.cs#L644-L681
   * https://docs.microsoft.com/en-us/typography/legacy/legacy_arabic_fonts */
  enum font_page_t
  {
    FONT_PAGE_NONE		= 0,
    FONT_PAGE_HEBREW		= 0xB100, /* Hebrew Windows 3.1 font page */
    FONT_PAGE_SIMP_ARABIC	= 0xB200, /* Simplified Arabic Windows 3.1 font page */
    FONT_PAGE_TRAD_ARABIC	= 0xB300, /* Traditional Arabic Windows 3.1 font page */
    FONT_PAGE_OEM_ARABIC	= 0xB400, /* OEM Arabic Windows 3.1 font page */
    FONT_PAGE_SIMP_FARSI	= 0xBA00, /* Simplified Farsi Windows 3.1 font page */
    FONT_PAGE_TRAD_FARSI	= 0xBB00, /* Traditional Farsi Windows 3.1 font page */
    FONT_PAGE_THAI		= 0xDE00  /* Thai Windows 3.1 font page */
  };
  font_page_t get_font_page () const
  { return (font_page_t) (version == 0 ? fsSelection & 0xFF00 : 0); }

  unsigned get_size () const
  {
    unsigned result = min_size;
    if (version >= 1) result += v1X.get_size ();
    if (version >= 2) result += v2X.get_size ();
    if (version >= 5) result += v5X.get_size ();
    return result;
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!c->check_struct (this))) return_trace (false);
    hb_barrier ();
    if (unlikely (version >= 1 && !v1X.sanitize (c))) return_trace (false);
    if (unlikely (version >= 2 && !v2X.sanitize (c))) return_trace (false);
    if (unlikely (version >= 5 && !v5X.sanitize (c))) return_trace (false);
    return_trace (true);
  }

  public:
  HBUINT16	version;
  HBINT16	xAvgCharWidth;
  HBUINT16	usWeightClass;
  HBUINT16	usWidthClass;
  HBUINT16	fsType;
  HBINT16	ySubscriptXSize;
  HBINT16	ySubscriptYSize;
  HBINT16	ySubscriptXOffset;
  HBINT16	ySubscriptYOffset;
  HBINT16	ySuperscriptXSize;
  HBINT16	ySuperscriptYSize;
  HBINT16	ySuperscriptXOffset;
  HBINT16	ySuperscriptYOffset;
  HBINT16	yStrikeoutSize;
  HBINT16	yStrikeoutPosition;
  HBINT16	sFamilyClass;
  HBUINT8	panose[10];
  HBUINT32	ulUnicodeRange[4];
  Tag		achVendID;
  HBUINT16	fsSelection;
  HBUINT16	usFirstCharIndex;
  HBUINT16	usLastCharIndex;
  HBINT16	sTypoAscender;
  HBINT16	sTypoDescender;
  HBINT16	sTypoLineGap;
  HBUINT16	usWinAscent;
  HBUINT16	usWinDescent;
  OS2V1Tail	v1X;
  OS2V2Tail	v2X;
  OS2V5Tail	v5X;
  public:
  DEFINE_SIZE_MIN (78);
};

} /* namespace OT */


#endif /* HB_OT_OS2_TABLE_HH */
