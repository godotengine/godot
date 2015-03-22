/* FriBidi
 * fribidi-char-sets-cp1256.c - CP1256 character set conversion routines
 *
 * $Id: fribidi-char-sets-cp1256.c,v 1.2 2004-05-03 22:05:19 behdad Exp $
 * $Author: behdad $
 * $Date: 2004-05-03 22:05:19 $
 * $Revision: 1.2 $
 * $Source: /home/behdad/src/fdo/fribidi/togit/git/../fribidi/fribidi2/charset/fribidi-char-sets-cp1256.c,v $
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

#include <fribidi-char-sets-cp1256.h>

#define ISO_HAMZA		0xc1
#define CP1256_DAD		0xD6

#define UNI_HAMZA		0x0621
#define UNI_DAD			0x0636

static FriBidiChar fribidi_cp1256_to_unicode_tab[] = {	/* 0x80-0xFF */
  0x20AC, 0x067E, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021,
  0x02C6, 0x2030, 0x0679, 0x2039, 0x0152, 0x0686, 0x0698, 0x0688,
  0x06AF, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
  0x06A9, 0x2122, 0x0691, 0x203A, 0x0153, 0x200C, 0x200D, 0x06BA,
  0x00A0, 0x060C, 0x00A2, 0x00A3, 0x00A4, 0x00A5, 0x00A6, 0x00A7,
  0x00A8, 0x00A9, 0x06BE, 0x00AB, 0x00AC, 0x00AD, 0x00AE, 0x00AF,
  0x00B0, 0x00B1, 0x00B2, 0x00B3, 0x00B4, 0x00B5, 0x00B6, 0x00B7,
  0x00B8, 0x00B9, 0x061B, 0x00BB, 0x00BC, 0x00BD, 0x00BE, 0x061F,
  0x06C1, 0x0621, 0x0622, 0x0623, 0x0624, 0x0625, 0x0626, 0x0627,
  0x0628, 0x0629, 0x062A, 0x062B, 0x062C, 0x062D, 0x062E, 0x062F,
  0x0630, 0x0631, 0x0632, 0x0633, 0x0634, 0x0635, 0x0636, 0x00D7,
  0x0637, 0x0638, 0x0639, 0x063A, 0x0640, 0x0641, 0x0642, 0x0643,
  0x00E0, 0x0644, 0x00E2, 0x0645, 0x0646, 0x0647, 0x0648, 0x00E7,
  0x00E8, 0x00E9, 0x00EA, 0x00EB, 0x0649, 0x064A, 0x00EE, 0x00EF,
  0x064B, 0x064C, 0x064D, 0x064E, 0x00F4, 0x064F, 0x0650, 0x00F7,
  0x0651, 0x00F9, 0x0652, 0x00FB, 0x00FC, 0x200E, 0x200F, 0x00ff
};

FriBidiChar
fribidi_cp1256_to_unicode_c (
  /* input */
  char sch
)
{
  register unsigned char ch = (unsigned char) sch;
  if (ch >= 0x80)
    return fribidi_cp1256_to_unicode_tab[ch - 0x80];
  else
    return ch;
}

char
fribidi_unicode_to_cp1256_c (
  /* input */
  FriBidiChar uch
)
{
  if (uch < 256)
    return (char) uch;
  if (uch >= UNI_HAMZA && uch <= UNI_DAD)
    return (char) (uch - UNI_HAMZA + ISO_HAMZA);
  else
    switch (uch)
      {
      case 0x0152:
	return (char) 0x8c;
      case 0x0153:
	return (char) 0x9c;
      case 0x0192:
	return (char) 0x83;
      case 0x02C6:
	return (char) 0x88;
      case 0x060C:
	return (char) 0xA1;
      case 0x061B:
	return (char) 0xBA;
      case 0x061F:
	return (char) 0xBF;
      case 0x0637:
	return (char) 0xD8;
      case 0x0638:
	return (char) 0xD9;
      case 0x0639:
	return (char) 0xDA;
      case 0x063A:
	return (char) 0xDB;
      case 0x0640:
	return (char) 0xDC;
      case 0x0641:
	return (char) 0xDD;
      case 0x0642:
	return (char) 0xDE;
      case 0x0643:
	return (char) 0xDF;
      case 0x0644:
	return (char) 0xE1;
      case 0x0645:
	return (char) 0xE3;
      case 0x0646:
	return (char) 0xE4;
      case 0x0647:
	return (char) 0xE5;
      case 0x0648:
	return (char) 0xE6;
      case 0x0649:
	return (char) 0xEC;
      case 0x064A:
	return (char) 0xED;
      case 0x064B:
	return (char) 0xF0;
      case 0x064C:
	return (char) 0xF1;
      case 0x064D:
	return (char) 0xF2;
      case 0x064E:
	return (char) 0xF3;
      case 0x064F:
	return (char) 0xF5;
      case 0x0650:
	return (char) 0xF6;
      case 0x0651:
	return (char) 0xF8;
      case 0x0652:
	return (char) 0xFA;
      case 0x0679:
	return (char) 0x8A;
      case 0x067E:
	return (char) 0x81;
      case 0x0686:
	return (char) 0x8D;
      case 0x0688:
	return (char) 0x8F;
      case 0x0691:
	return (char) 0x9A;
      case 0x0698:
	return (char) 0x8E;
      case 0x06A9:
	return (char) 0x98;
      case 0x06AF:
	return (char) 0x90;
      case 0x06BA:
	return (char) 0x9F;
      case 0x06BE:
	return (char) 0xAA;
      case 0x06C1:
	return (char) 0xC0;
      case 0x200C:
	return (char) 0x9D;
      case 0x200D:
	return (char) 0x9E;
      case 0x200E:
	return (char) 0xFD;
      case 0x200F:
	return (char) 0xFE;
      case 0x2013:
	return (char) 0x96;
      case 0x2014:
	return (char) 0x97;
      case 0x2018:
	return (char) 0x91;
      case 0x2019:
	return (char) 0x92;
      case 0x201A:
	return (char) 0x82;
      case 0x201C:
	return (char) 0x93;
      case 0x201D:
	return (char) 0x94;
      case 0x201E:
	return (char) 0x84;
      case 0x2020:
	return (char) 0x86;
      case 0x2021:
	return (char) 0x87;
      case 0x2022:
	return (char) 0x95;
      case 0x2026:
	return (char) 0x85;
      case 0x2030:
	return (char) 0x89;
      case 0x2039:
	return (char) 0x8B;
      case 0x203A:
	return (char) 0x9B;
      case 0x20AC:
	return (char) 0x80;
      case 0x2122:
	return (char) 0x99;

      default:
	return '?';
      }
}

/* Editor directions:
 * vim:textwidth=78:tabstop=8:shiftwidth=2:autoindent:cindent
 */
