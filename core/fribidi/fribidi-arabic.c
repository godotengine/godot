/* fribidi-arabic.c - Arabic shaping
 *
 * Copyright (C) 2005  Behdad Esfahbod
 *
 * This file is part of GNU FriBidi.
 * 
 * GNU FriBidi is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 * 
 * GNU FriBidi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with GNU FriBidi; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 * 
 * For licensing issues, contact <license@farsiweb.info> or write to
 * Sharif FarsiWeb, Inc., PO Box 13445-389, Tehran, Iran.
 */
/* $Id: fribidi-arabic.c,v 1.3 2007-04-05 16:14:39 behdad Exp $
 * $Author: behdad $
 * $Date: 2007-04-05 16:14:39 $
 * $Revision: 1.3 $
 * $Source: /home/behdad/src/fdo/fribidi/togit/git/../fribidi/fribidi2/lib/fribidi-arabic.c,v $
 *
 * Author(s):
 *   Behdad Esfahbod, 2005
 */

#include "common.h"

#if HAVE_STDLIB_H+0
# include <stdlib.h>
#endif


#include <fribidi-arabic.h>
#include <fribidi-unicode.h>


typedef struct _PairMap {
  FriBidiChar pair[2], to;
} PairMap;


#define FRIBIDI_ACCESS_SHAPE_TABLE(table,min,max,x,shape) (table), (min), (max)
# define FRIBIDI_ACCESS_SHAPE_TABLE_REAL(table,min,max,x,shape) \
	(((x)<(min)||(x)>(max))?(x):(table)[(x)-(min)][(shape)])

#include "arabic-shaping.tab.i"
#include "arabic-misc.tab.i"


static void
fribidi_shape_arabic_joining (
  /* input */
  const FriBidiChar table[][4],
  FriBidiChar min,
  FriBidiChar max,
  const FriBidiStrIndex len,
  const FriBidiArabicProp *ar_props,
  /* input and output */
  FriBidiChar *str
)
{
  register FriBidiStrIndex i;

  for (i = 0; i < len; i++)
    if (FRIBIDI_ARAB_SHAPES(ar_props[i]))
      str[i] = FRIBIDI_ACCESS_SHAPE_TABLE_REAL (table, min, max, str[i], FRIBIDI_JOIN_SHAPE (ar_props[i]));
}



static int
comp_PairMap (const void *pa, const void *pb)
{
  PairMap *a = (PairMap *)pa;
  PairMap *b = (PairMap *)pb;

  if (a->pair[0] != b->pair[0])
    return a->pair[0] < b->pair[0] ? -1 : +1;
  else
    return a->pair[1] < b->pair[1] ? -1 :
           a->pair[1] > b->pair[1] ? +1 :
	   0;
}


static FriBidiChar
find_pair_match (const PairMap *table, int size, FriBidiChar first, FriBidiChar second)
{
  PairMap *match;
  PairMap x;
  x.pair[0] = first;
  x.pair[1] = second;
  x.to = 0;
  match = bsearch (&x, table, size, sizeof (table[0]), comp_PairMap);
  return match ? match->to : 0;
}

#define PAIR_MATCH(table,len,first,second) \
	((first)<(table[0].pair[0])||(first)>(table[len-1].pair[0])?0: \
	 find_pair_match(table, len, first, second))

static void
fribidi_shape_arabic_ligature (
  /* input */
  const PairMap *table,
  int size,
  const FriBidiLevel *embedding_levels,
  const FriBidiStrIndex len,
  /* input and output */
  FriBidiArabicProp *ar_props,
  FriBidiChar *str
)
{
  /* TODO: This doesn't form ligatures for even-level Arabic text.
   * no big problem though. */
  register FriBidiStrIndex i;

  for (i = 0; i < len - 1; i++) {
    register FriBidiChar c;
    if (FRIBIDI_LEVEL_IS_RTL(embedding_levels[i]) &&
	embedding_levels[i] == embedding_levels[i+1] &&
	(c = PAIR_MATCH(table, size, str[i], str[i+1])))
      {
	str[i] = FRIBIDI_CHAR_FILL;
	FRIBIDI_SET_BITS(ar_props[i], FRIBIDI_MASK_LIGATURED);
	str[i+1] = c;
      }
  }
}

#define DO_LIGATURING(table, levels, len, ar_props, str) \
	fribidi_shape_arabic_ligature ((table), sizeof(table)/sizeof((table)[0]), levels, len, ar_props, str)

#define DO_SHAPING(tablemacro, len, ar_props, str) \
	fribidi_shape_arabic_joining (tablemacro(,), len, ar_props, str);
	



FRIBIDI_ENTRY void
fribidi_shape_arabic (
  /* input */
  FriBidiFlags flags,
  const FriBidiLevel *embedding_levels,
  const FriBidiStrIndex len,
  /* input and output */
  FriBidiArabicProp *ar_props,
  FriBidiChar *str
)
{
  DBG ("in fribidi_shape_arabic");

  if UNLIKELY
    (len == 0 || !str) return;

  DBG ("in fribidi_shape");

  fribidi_assert (ar_props);

  if (FRIBIDI_TEST_BITS (flags, FRIBIDI_FLAG_SHAPE_ARAB_PRES))
    {
      DO_SHAPING (FRIBIDI_GET_ARABIC_SHAPE_PRES, len, ar_props, str);
    }

  if (FRIBIDI_TEST_BITS (flags, FRIBIDI_FLAG_SHAPE_ARAB_LIGA))
    {
      DO_LIGATURING (mandatory_liga_table, embedding_levels, len, ar_props, str);
    }

  if (FRIBIDI_TEST_BITS (flags, FRIBIDI_FLAG_SHAPE_ARAB_CONSOLE))
    {
      DO_LIGATURING (console_liga_table, embedding_levels, len, ar_props, str);
      DO_SHAPING (FRIBIDI_GET_ARABIC_SHAPE_NSM, len, ar_props, str);
    }
}

/* Editor directions:
 * Local Variables:
 *   mode: c
 *   c-basic-offset: 2
 *   indent-tabs-mode: t
 *   tab-width: 8
 * End:
 * vim: textwidth=78: autoindent: cindent: shiftwidth=2: tabstop=8:
 */
