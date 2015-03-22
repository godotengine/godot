/* FriBidi
 * fribidi-shape.c - shaping
 *
 * $Id: fribidi-shape.c,v 1.1 2005-11-03 01:39:01 behdad Exp $
 * $Author: behdad $
 * $Date: 2005-11-03 01:39:01 $
 * $Revision: 1.1 $
 * $Source: /home/behdad/src/fdo/fribidi/togit/git/../fribidi/fribidi2/lib/fribidi-shape.c,v $
 *
 * Authors:
 *   Behdad Esfahbod, 2001, 2002, 2004
 *   Dov Grobgeld, 1999, 2000
 *
 * Copyright (C) 2005 Behdad Esfahbod
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

#include "common.h"

#include <fribidi-shape.h>
#include <fribidi-mirroring.h>
#include <fribidi-arabic.h>


FRIBIDI_ENTRY void
fribidi_shape (
  /* input */
  FriBidiFlags flags,
  const FriBidiLevel *embedding_levels,
  const FriBidiStrIndex len,
  /* input and output */
  FriBidiArabicProp *ar_props,
  FriBidiChar *str
)
{
  if UNLIKELY
    (len == 0 || !str) return;

  DBG ("in fribidi_shape");

  fribidi_assert (embedding_levels);

  if (ar_props)
    fribidi_shape_arabic (flags, embedding_levels, len, ar_props, str);

  if (FRIBIDI_TEST_BITS (flags, FRIBIDI_FLAG_SHAPE_MIRRORING))
    fribidi_shape_mirroring (embedding_levels, len, str);
}


/* Editor directions:
 * vim:textwidth=78:tabstop=8:shiftwidth=2:autoindent:cindent
 */
