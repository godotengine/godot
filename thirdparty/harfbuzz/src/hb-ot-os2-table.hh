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


namespace OT {

/*
 * OS/2 and Windows Metrics
 * http://www.microsoft.com/typography/otspec/os2.htm
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

  public:
  HBUINT16	version;

  /* Version 0 */
  HBINT16		xAvgCharWidth;
  HBUINT16	usWeightClass;
  HBUINT16	usWidthClass;
  HBUINT16	fsType;
  HBINT16		ySubscriptXSize;
  HBINT16		ySubscriptYSize;
  HBINT16		ySubscriptXOffset;
  HBINT16		ySubscriptYOffset;
  HBINT16		ySuperscriptXSize;
  HBINT16		ySuperscriptYSize;
  HBINT16		ySuperscriptXOffset;
  HBINT16		ySuperscriptYOffset;
  HBINT16		yStrikeoutSize;
  HBINT16		yStrikeoutPosition;
  HBINT16		sFamilyClass;
  HBUINT8		panose[10];
  HBUINT32		ulUnicodeRange[4];
  Tag		achVendID;
  HBUINT16	fsSelection;
  HBUINT16	usFirstCharIndex;
  HBUINT16	usLastCharIndex;
  HBINT16		sTypoAscender;
  HBINT16		sTypoDescender;
  HBINT16		sTypoLineGap;
  HBUINT16	usWinAscent;
  HBUINT16	usWinDescent;

  /* Version 1 */
  //HBUINT32 ulCodePageRange1;
  //HBUINT32 ulCodePageRange2;

  /* Version 2 */
  //HBINT16 sxHeight;
  //HBINT16 sCapHeight;
  //HBUINT16  usDefaultChar;
  //HBUINT16  usBreakChar;
  //HBUINT16  usMaxContext;

  /* Version 5 */
  //HBUINT16  usLowerOpticalPointSize;
  //HBUINT16  usUpperOpticalPointSize;

  public:
  DEFINE_SIZE_STATIC (78);
};

} /* namespace OT */


#endif /* HB_OT_OS2_TABLE_HH */
