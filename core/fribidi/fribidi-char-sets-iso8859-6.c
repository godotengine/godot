/* FriBidi
 * fribidi-char-sets-iso8859-6.c - ISO8859-6 character set conversion routines
 *
 * $Id: fribidi-char-sets-iso8859-6.c,v 1.2 2004-05-03 22:05:19 behdad Exp $
 * $Author: behdad $
 * $Date: 2004-05-03 22:05:19 $
 * $Revision: 1.2 $
 * $Source: /home/behdad/src/fdo/fribidi/togit/git/../fribidi/fribidi2/charset/fribidi-char-sets-iso8859-6.c,v $
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

#include <fribidi-char-sets-iso8859-6.h>

#define ISO_HAMZA		0xc1
#define ISO_SUKUN		0xf2

#define UNI_HAMZA		0x0621
#define UNI_SUKUN		0x0652

FriBidiChar
fribidi_iso8859_6_to_unicode_c (
  /* input */
  char sch
)
{
  register unsigned char ch = (unsigned char) sch;
  if (ch >= ISO_HAMZA && ch <= ISO_SUKUN)
    return ch - ISO_HAMZA + UNI_HAMZA;
  else
    return ch;
}

char
fribidi_unicode_to_iso8859_6_c (
  /* input */
  FriBidiChar uch
)
{
  if (uch >= UNI_HAMZA && uch <= UNI_SUKUN)
    return (char) (uch - UNI_HAMZA + ISO_HAMZA);
  else if (uch < 256)
    return (char) uch;
  else if (uch == 0x060c)
    return (char) 0xac;
  else if (uch == 0x061b)
    return (char) 0xbb;
  else if (uch == 0x061f)
    return (char) 0xbf;
  else
    return '?';
}

/* Editor directions:
 * vim:textwidth=78:tabstop=8:shiftwidth=2:autoindent:cindent
 */
