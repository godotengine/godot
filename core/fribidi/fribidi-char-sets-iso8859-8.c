/* FriBidi
 * fribidi-char-sets-iso8859-8.c - ISO8859-8 character set conversion routines
 *
 * $Id: fribidi-char-sets-iso8859-8.c,v 1.2 2004-05-03 22:05:19 behdad Exp $
 * $Author: behdad $
 * $Date: 2004-05-03 22:05:19 $
 * $Revision: 1.2 $
 * $Source: /home/behdad/src/fdo/fribidi/togit/git/../fribidi/fribidi2/charset/fribidi-char-sets-iso8859-8.c,v $
 *
 * Authors:
 *   Behdad Esfahbod, 2001, 2002, 2004
 *   Dov Grobgeld, 1999, 2000
 *
 * Copyright (C) 2004 Sharif FarsiWeb, Inc
 * Copyright (C) 2001,2002 Behdad Esfahbod
 * Copyright (C) 1999,2000 Dov Grobgeld
 * 
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library, in a file named COPYING; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA 02110-1301, USA
 * 
 * For licensing issues, contact <license@farsiweb.info>.
 */

#include <common.h>

#include <fribidi-char-sets-iso8859-8.h>

#include <fribidi-unicode.h>

/* The following are proposed extensions to ISO8859-8. */
#define ISO_8859_8_LRM		0xFD
#define ISO_8859_8_RLM		0xFE
#define ISO_8859_8_LRE		0xFB
#define ISO_8859_8_RLE		0xFC
#define ISO_8859_8_PDF		0xDD
#define ISO_8859_8_LRO		0xDB
#define ISO_8859_8_RLO		0xDC
#define ISO_ALEF		0xE0
#define ISO_TAV			0xFA

#define UNI_ALEF		0x05D0
#define UNI_TAV			0x05EA

FriBidiChar
fribidi_iso8859_8_to_unicode_c (
  /* input */
  char sch
)
{
  register unsigned char ch = (unsigned char) sch;
  if (ch < ISO_8859_8_LRO)
    return ch;
  else if (ch >= ISO_ALEF && ch <= ISO_TAV)
    return ch - ISO_ALEF + UNI_ALEF;
  switch (ch)
    {
    case ISO_8859_8_RLM:
      return FRIBIDI_CHAR_RLM;
    case ISO_8859_8_LRM:
      return FRIBIDI_CHAR_LRM;
    case ISO_8859_8_RLO:
      return FRIBIDI_CHAR_RLO;
    case ISO_8859_8_LRO:
      return FRIBIDI_CHAR_LRO;
    case ISO_8859_8_RLE:
      return FRIBIDI_CHAR_RLE;
    case ISO_8859_8_LRE:
      return FRIBIDI_CHAR_LRE;
    case ISO_8859_8_PDF:
      return FRIBIDI_CHAR_PDF;
    default:
      return '?';
    }
}

char
fribidi_unicode_to_iso8859_8_c (
  /* input */
  FriBidiChar uch
)
{
  if (uch < 128)
    return (char) uch;
  if (uch >= UNI_ALEF && uch <= UNI_TAV)
    return (char) (uch - UNI_ALEF + ISO_ALEF);
  switch (uch)
    {
    case FRIBIDI_CHAR_RLM:
      return (char) ISO_8859_8_RLM;
    case FRIBIDI_CHAR_LRM:
      return (char) ISO_8859_8_LRM;
    case FRIBIDI_CHAR_RLO:
      return (char) ISO_8859_8_RLO;
    case FRIBIDI_CHAR_LRO:
      return (char) ISO_8859_8_LRO;
    case FRIBIDI_CHAR_RLE:
      return (char) ISO_8859_8_RLE;
    case FRIBIDI_CHAR_LRE:
      return (char) ISO_8859_8_LRE;
    case FRIBIDI_CHAR_PDF:
      return (char) ISO_8859_8_PDF;
    }
  return '?';
}

/* Editor directions:
 * vim:textwidth=78:tabstop=8:shiftwidth=2:autoindent:cindent
 */
