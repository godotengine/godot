/* FriBidi
 * fribidi-char-sets-utf8.c - UTF-8 character set conversion routines
 *
 * $Id: fribidi-char-sets-utf8.c,v 1.3 2005-07-30 09:06:28 behdad Exp $
 * $Author: behdad $
 * $Date: 2005-07-30 09:06:28 $
 * $Revision: 1.3 $
 * $Source: /home/behdad/src/fdo/fribidi/togit/git/../fribidi/fribidi2/charset/fribidi-char-sets-utf8.c,v $
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

#include <fribidi-char-sets-utf8.h>

#include <fribidi-unicode.h>

FriBidiStrIndex
fribidi_utf8_to_unicode (
  /* input */
  const char *ss,
  FriBidiStrIndex len,
  /* output */
  FriBidiChar *us
)
{
  FriBidiStrIndex length;
  const unsigned char *s = (unsigned const char *) ss;
  const unsigned char *t = s;

  length = 0;
  while ((FriBidiStrIndex) (s - t) < len)
    {
      register unsigned char ch = *s;
      if (ch <= 0x7f)		/* one byte */
	{
	  *us++ = *s++;
	}
      else if (ch <= 0xdf)	/* 2 byte */
	{
	  *us++ = ((*s & 0x1f) << 6) + (*(s + 1) & 0x3f);
	  s += 2;
	}
      else			/* 3 byte */
	{
	  *us++ =
	    ((int) (*s & 0x0f) << 12) +
	    ((*(s + 1) & 0x3f) << 6) + (*(s + 2) & 0x3f);
	  s += 3;
	}
      length++;
    }
  return (length);
}

FriBidiStrIndex
fribidi_unicode_to_utf8 (
  /* input */
  const FriBidiChar *us,
  FriBidiStrIndex len,
  /* output */
  char *ss
)
{
  FriBidiStrIndex i;
  unsigned char *s = (unsigned char *) ss;
  unsigned char *t = s;

  for (i = 0; i < len; i++)
    {
      FriBidiChar mychar = us[i];
      if (mychar <= 0x7F)
	{			/* 7 sig bits */
	  *t++ = mychar;
	}
      else if (mychar <= 0x7FF)
	{			/* 11 sig bits */
	  *t++ = 0xC0 | (unsigned char) (mychar >> 6);	/* upper 5 bits */
	  *t++ = 0x80 | (unsigned char) (mychar & 0x3F);	/* lower 6 bits */
	}
      else if (mychar <= 0xFFFF)
	{			/* 16 sig bits */
	  *t++ = 0xE0 | (unsigned char) (mychar >> 12);	/* upper 4 bits */
	  *t++ = 0x80 | (unsigned char) ((mychar >> 6) & 0x3F);	/* next 6 bits */
	  *t++ = 0x80 | (unsigned char) (mychar & 0x3F);	/* lowest 6 bits */
	}
      else if (mychar < FRIBIDI_UNICODE_CHARS)
	{			/* 21 sig bits */
	  *t++ = 0xF0 | (unsigned char) ((mychar >> 18) & 0x07);	/* upper 3 bits */
	  *t++ = 0x80 | (unsigned char) ((mychar >> 12) & 0x3F);	/* next 6 bits */
	  *t++ = 0x80 | (unsigned char) ((mychar >> 6) & 0x3F);	/* next 6 bits */
	  *t++ = 0x80 | (unsigned char) (mychar & 0x3F);	/* lowest 6 bits */
	}
    }
  *t = 0;

  return (t - s);
}

/* Editor directions:
 * vim:textwidth=78:tabstop=8:shiftwidth=2:autoindent:cindent
 */
