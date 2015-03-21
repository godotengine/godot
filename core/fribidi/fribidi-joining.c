/* FriBidi
 * fribidi-joining.h - Arabic joining algorithm
 *
 * $Id: fribidi-joining.c,v 1.6 2006-01-31 03:23:13 behdad Exp $
 * $Author: behdad $
 * $Date: 2006-01-31 03:23:13 $
 * $Revision: 1.6 $
 * $Source: /home/behdad/src/fdo/fribidi/togit/git/../fribidi/fribidi2/lib/fribidi-joining.c,v $
 *
 * Authors:
 *   Behdad Esfahbod, 2004
 *
 * Copyright (C) 2004 Sharif FarsiWeb, Inc
 * Copyright (C) 2004 Behdad Esfahbod
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

#include <fribidi-joining.h>

#include "mem.h"
#include "bidi-types.h"
#include "joining-types.h"

#if DEBUG+0
/*======================================================================
 *  For debugging, define some functions for printing joining types and
 *  properties.
 *----------------------------------------------------------------------*/

static void
print_joining_types (
  /* input */
  const FriBidiLevel *embedding_levels,
  const FriBidiStrIndex len,
  const FriBidiJoiningType *jtypes
)
{
  register FriBidiStrIndex i;

  fribidi_assert (jtypes);

  MSG ("  Join. types: ");
  for (i = 0; i < len; i++)
    MSG2 ("%c", fribidi_char_from_joining_type (jtypes[i],
						!FRIBIDI_LEVEL_IS_RTL
						(embedding_levels[i])));
  MSG ("\n");
}
#endif /* DEBUG */

#define FRIBIDI_CONSISTENT_LEVEL(i)	\
	(FRIBIDI_IS_EXPLICIT_OR_BN (bidi_types[(i)])	\
	 ? FRIBIDI_SENTINEL	\
	 : embedding_levels[(i)])

#define FRIBIDI_LEVELS_MATCH(i, j)	\
	((i) == (j) || (i) == FRIBIDI_SENTINEL || (j) == FRIBIDI_SENTINEL)

FRIBIDI_ENTRY void
fribidi_join_arabic (
  /* input */
  const FriBidiCharType *bidi_types,
  const FriBidiStrIndex len,
  const FriBidiLevel *embedding_levels,
  /* input and output */
  FriBidiArabicProp *ar_props
)
{
  if UNLIKELY
    (len == 0) return;

  DBG ("in fribidi_join_arabic");

  fribidi_assert (bidi_types);
  fribidi_assert (embedding_levels);
  fribidi_assert (ar_props);

# if DEBUG
  if UNLIKELY
    (fribidi_debug_status ())
    {
      print_joining_types (embedding_levels, len, ar_props);
    }
# endif	/* DEBUG */

  /* The joining algorithm turned out very very dirty :(.  That's what happens
   * when you follow the standard which has never been implemented closely
   * before.
   */

  /* 8.2 Arabic - Cursive Joining */
  DBG ("Arabic cursive joining");
  {
    /* The following do not need to be initialized as long as joins is
     * initialized to false.  We just do to turn off compiler warnings. */
    register FriBidiStrIndex saved = 0;
    register FriBidiLevel saved_level = FRIBIDI_SENTINEL;
    register fribidi_boolean saved_shapes = false;
    register FriBidiArabicProp saved_joins_following_mask = 0;

    register fribidi_boolean joins = false;
    register FriBidiStrIndex i;

    for (i = 0; i < len; i++)
      if (!FRIBIDI_IS_JOINING_TYPE_G (ar_props[i]))
	{
	  register fribidi_boolean disjoin = false;
	  register fribidi_boolean shapes = FRIBIDI_ARAB_SHAPES (ar_props[i]);
	  register FriBidiLevel level = FRIBIDI_CONSISTENT_LEVEL (i);

	  if (joins && !FRIBIDI_LEVELS_MATCH (saved_level, level))
	    {
	      disjoin = true;
	      joins = false;
	    }

	  if (!FRIBIDI_IS_JOIN_SKIPPED (ar_props[i]))
	    {
	      register const FriBidiArabicProp joins_preceding_mask =
		FRIBIDI_JOINS_PRECEDING_MASK (level);

	      if (!joins)
		{
		  if (shapes)
		    FRIBIDI_UNSET_BITS (ar_props[i], joins_preceding_mask);
		}
	      else if (!FRIBIDI_TEST_BITS (ar_props[i], joins_preceding_mask))
	        {
		  disjoin = true;
		}
	      else
	        {
		  register FriBidiStrIndex j;
		  /* This is a FriBidi extension:  we set joining properties
		   * for skipped characters in between, so we can put NSMs on tatweel
		   * later if we want.  Useful on console for example.
		   */
		  for (j = saved + 1; j < i; j++)
		    FRIBIDI_SET_BITS (ar_props[j], joins_preceding_mask | saved_joins_following_mask);
		}
	    }

	  if (disjoin && saved_shapes)
	    FRIBIDI_UNSET_BITS (ar_props[saved], saved_joins_following_mask);

	  if (!FRIBIDI_IS_JOIN_SKIPPED (ar_props[i]))
	    {
	      saved = i;
	      saved_level = level;
	      saved_shapes = shapes;
	      saved_joins_following_mask =
		FRIBIDI_JOINS_FOLLOWING_MASK (level);
	      joins =
		FRIBIDI_TEST_BITS (ar_props[i], saved_joins_following_mask);
	    }
	}
    if ((joins) && saved_shapes)
      FRIBIDI_UNSET_BITS (ar_props[saved], saved_joins_following_mask);

  }

# if DEBUG
  if UNLIKELY
    (fribidi_debug_status ())
    {
      print_joining_types (embedding_levels, len, ar_props);
    }
# endif	/* DEBUG */

  DBG ("leaving fribidi_join_arabic");
}

/* Editor directions:
 * vim:textwidth=78:tabstop=8:shiftwidth=2:autoindent:cindent
 */
