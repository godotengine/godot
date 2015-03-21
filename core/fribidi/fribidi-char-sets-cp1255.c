/* FriBidi
 * fribidi-char-sets-cp1255.c - CP1255 character set conversion routines
 *
 * $Id: fribidi-char-sets-cp1255.c,v 1.3 2008-04-22 19:50:46 behdad Exp $
 * $Author: behdad $
 * $Date: 2008-04-22 19:50:46 $
 * $Revision: 1.3 $
 * $Source: /home/behdad/src/fdo/fribidi/togit/git/../fribidi/fribidi2/charset/fribidi-char-sets-cp1255.c,v $
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

#include <fribidi-char-sets-cp1255.h>

#define ISO_ALEF		0xE0
#define ISO_TAV			0xFA
#define CP1255_SHEVA		0xC0
#define CP1255_SOF_PASUQ	0xD3
#define CP1255_DOUBLE_VAV	0xD4
#define CP1255_GERSHAYIM	0xD8
#define CP1255_LRM		0xFD
#define CP1255_RLM		0xFE

#define UNI_ALEF		0x05D0
#define UNI_TAV			0x05EA
#define UNI_SHEVA		0x05B0
#define UNI_SOF_PASUQ		0x05C3
#define UNI_DOUBLE_VAV		0x05F0
#define UNI_GERSHAYIM		0x05F4
#define UNI_LRM			0x200E
#define UNI_RLM			0x200F

static FriBidiChar fribidi_cp1255_to_unicode_tab[] = {	/* 0x80-0xBF */
  0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021,
  0x02C6, 0x2030, 0x008a, 0x2039, 0x008c, 0x008d, 0x008e, 0x008f,
  0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
  0x02DC, 0x2122, 0x009a, 0x203A, 0x009c, 0x009d, 0x009e, 0x009f,
  0x00A0, 0x00A1, 0x00A2, 0x00A3, 0x20AA, 0x00A5, 0x00A6, 0x00A7,
  0x00A8, 0x00A9, 0x00D7, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x00AF,
  0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x00B4, 0x00B5, 0x00B6, 0x00B7,
  0x00B8, 0x00B9, 0x00F7, 0x00BB, 0x00BC, 0x00BD, 0x00BE, 0x00BF
};

FriBidiChar
fribidi_cp1255_to_unicode_c (
  /* input */
  char sch
)
{
  register unsigned char ch = (unsigned char) sch;
  if (ch >= ISO_ALEF && ch <= ISO_TAV)
    return ch - ISO_ALEF + UNI_ALEF;
  else if (ch >= CP1255_SHEVA && ch <= CP1255_SOF_PASUQ)
    return ch - CP1255_SHEVA + UNI_SHEVA;
  else if (ch >= CP1255_DOUBLE_VAV && ch <= CP1255_GERSHAYIM)
    return ch - CP1255_DOUBLE_VAV + UNI_DOUBLE_VAV;
  /* cp1256 specific chars */
  else if (ch >= 0x80 && ch <= 0xbf)
    return fribidi_cp1255_to_unicode_tab[ch - 0x80];
  else if (ch == CP1255_LRM || ch == CP1255_RLM) 
    return ch - CP1255_LRM + UNI_LRM;
    /* treat LRM/LRM charrectes correctly */
  else 
    return ch;
}

char
fribidi_unicode_to_cp1255_c (
  /* input */
  FriBidiChar uch
)
{
  if (uch >= UNI_ALEF && uch <= UNI_TAV)
    return (char) (uch - UNI_ALEF + ISO_ALEF);
  if (uch >= UNI_SHEVA && uch <= UNI_SOF_PASUQ)
    return (char) (uch - UNI_SHEVA + CP1255_SHEVA);
  if (uch >= UNI_DOUBLE_VAV && uch <= UNI_GERSHAYIM)
    return (char) (uch - UNI_DOUBLE_VAV + CP1255_DOUBLE_VAV);
  /* TODO: handle pre-composed and presentation chars */
  if (uch == UNI_LRM || uch==UNI_RLM)
    return (char) (uch - UNI_LRM + CP1255_LRM);
    /* Treat LRM/RLM charrecters correctly */
  else if (uch < 256)
    return (char) uch;
  else
    return '?';
}

/* Editor directions:
 * vim:textwidth=78:tabstop=8:shiftwidth=2:autoindent:cindent
 */
