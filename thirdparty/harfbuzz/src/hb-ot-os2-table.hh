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
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_OT_OS2_TABLE_HH
#define HB_OT_OS2_TABLE_HH

#include "hb-open-type-private.hh"
#include "hb-ot-os2-unicode-ranges.hh"
#include "hb-subset-plan.hh"

namespace OT {

/*
 * OS/2 and Windows Metrics
 * https://docs.microsoft.com/en-us/typography/opentype/spec/os2
 */
#define HB_OT_TAG_os2 HB_TAG('O','S','/','2')

struct os2
{
  static const hb_tag_t tableTag = HB_OT_TAG_os2;

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  inline bool subset (hb_subset_plan_t *plan) const
  {
    hb_blob_t *os2_blob = hb_sanitize_context_t().reference_table<os2> (plan->source);
    hb_blob_t *os2_prime_blob = hb_blob_create_sub_blob (os2_blob, 0, -1);
    // TODO(grieger): move to hb_blob_copy_writable_or_fail
    hb_blob_destroy (os2_blob);

    os2 *os2_prime = (os2 *) hb_blob_get_data_writable (os2_prime_blob, nullptr);
    if (unlikely (!os2_prime)) {
      hb_blob_destroy (os2_prime_blob);
      return false;
    }

    uint16_t min_cp, max_cp;
    find_min_and_max_codepoint (plan->unicodes, &min_cp, &max_cp);
    os2_prime->usFirstCharIndex.set (min_cp);
    os2_prime->usLastCharIndex.set (max_cp);

    _update_unicode_ranges (plan->unicodes, os2_prime->ulUnicodeRange);
    bool result = plan->add_table (HB_OT_TAG_os2, os2_prime_blob);

    hb_blob_destroy (os2_prime_blob);
    return result;
  }

  inline void _update_unicode_ranges (const hb_set_t *codepoints,
                                      HBUINT32 ulUnicodeRange[4]) const
  {
    for (unsigned int i = 0; i < 4; i++)
      ulUnicodeRange[i].set (0);

    hb_codepoint_t cp = HB_SET_VALUE_INVALID;
    while (codepoints->next (&cp)) {
      unsigned int bit = hb_get_unicode_range_bit (cp);
      if (bit < 128)
      {
        unsigned int block = bit / 32;
        unsigned int bit_in_block = bit % 32;
        unsigned int mask = 1 << bit_in_block;
        ulUnicodeRange[block].set (ulUnicodeRange[block] | mask);
      }
      if (cp >= 0x10000 && cp <= 0x110000)
      {
        /* the spec says that bit 57 ("Non Plane 0") implies that there's
           at least one codepoint beyond the BMP; so I also include all
           the non-BMP codepoints here */
        ulUnicodeRange[1].set (ulUnicodeRange[1] | (1 << 25));
      }
    }
  }

  static inline void find_min_and_max_codepoint (const hb_set_t *codepoints,
                                                 uint16_t *min_cp, /* OUT */
                                                 uint16_t *max_cp  /* OUT */)
  {
    *min_cp = codepoints->get_min ();
    *max_cp = codepoints->get_max ();
  }

  enum font_page_t {
    HEBREW_FONT_PAGE		= 0xB100, // Hebrew Windows 3.1 font page
    SIMP_ARABIC_FONT_PAGE	= 0xB200, // Simplified Arabic Windows 3.1 font page
    TRAD_ARABIC_FONT_PAGE	= 0xB300, // Traditional Arabic Windows 3.1 font page
    OEM_ARABIC_FONT_PAGE	= 0xB400, // OEM Arabic Windows 3.1 font page
    SIMP_FARSI_FONT_PAGE	= 0xBA00, // Simplified Farsi Windows 3.1 font page
    TRAD_FARSI_FONT_PAGE	= 0xBB00, // Traditional Farsi Windows 3.1 font page
    THAI_FONT_PAGE		= 0xDE00  // Thai Windows 3.1 font page
  };

  // https://github.com/Microsoft/Font-Validator/blob/520aaae/OTFontFileVal/val_OS2.cs#L644-L681
  inline font_page_t get_font_page () const
  {
    if (version != 0)
      return (font_page_t) 0;
    return (font_page_t) (fsSelection & 0xFF00);
  }

  public:
  HBUINT16	version;

  /* Version 0 */
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

  /* Version 1 */
  //HBUINT32	ulCodePageRange1;
  //HBUINT32	ulCodePageRange2;

  /* Version 2 */
  //HBINT16	sxHeight;
  //HBINT16	sCapHeight;
  //HBUINT16	usDefaultChar;
  //HBUINT16	usBreakChar;
  //HBUINT16	usMaxContext;

  /* Version 5 */
  //HBUINT16	usLowerOpticalPointSize;
  //HBUINT16	usUpperOpticalPointSize;

  public:
  DEFINE_SIZE_STATIC (78);
};

} /* namespace OT */


#endif /* HB_OT_OS2_TABLE_HH */
