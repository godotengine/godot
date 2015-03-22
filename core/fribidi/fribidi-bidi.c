/* FriBidi
 * fribidi-bidi.c - bidirectional algorithm
 *
 * $Id: fribidi-bidi.c,v 1.21 2007-03-15 18:09:25 behdad Exp $
 * $Author: behdad $
 * $Date: 2007-03-15 18:09:25 $
 * $Revision: 1.21 $
 * $Source: /home/behdad/src/fdo/fribidi/togit/git/../fribidi/fribidi2/lib/fribidi-bidi.c,v $
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

#include "common.h"

#include <fribidi-bidi.h>
#include <fribidi-mirroring.h>
#include <fribidi-unicode.h>

#include "mem.h"
#include "bidi-types.h"
#include "run.h"

/*
 * This file implements most of Unicode Standard Annex #9, Tracking Number 13.
 */

#ifndef MAX
# define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif /* !MAX */

/* Some convenience macros */
#define RL_TYPE(list) ((list)->type)
#define RL_LEN(list) ((list)->len)
#define RL_POS(list) ((list)->pos)
#define RL_LEVEL(list) ((list)->level)

static FriBidiRun *
merge_with_prev (
  FriBidiRun *second
)
{
  FriBidiRun *first;

  fribidi_assert (second);
  fribidi_assert (second->next);
  first = second->prev;
  fribidi_assert (first);

  first->next = second->next;
  first->next->prev = first;
  RL_LEN (first) += RL_LEN (second);
  free_run (second);
  return first;
}

static void
compact_list (
  FriBidiRun *list
)
{
  fribidi_assert (list);

  if (list->next)
    for_run_list (list, list)
      if (RL_TYPE (list->prev) == RL_TYPE (list)
	  && RL_LEVEL (list->prev) == RL_LEVEL (list))
      list = merge_with_prev (list);
}

static void
compact_neutrals (
  FriBidiRun *list
)
{
  fribidi_assert (list);

  if (list->next)
    {
      for_run_list (list, list)
      {
	if (RL_LEVEL (list->prev) == RL_LEVEL (list)
	    &&
	    ((RL_TYPE
	      (list->prev) == RL_TYPE (list)
	      || (FRIBIDI_IS_NEUTRAL (RL_TYPE (list->prev))
		  && FRIBIDI_IS_NEUTRAL (RL_TYPE (list))))))
	  list = merge_with_prev (list);
      }
    }
}

#if DEBUG+0
/*======================================================================
 *  For debugging, define some functions for printing the types and the
 *  levels.
 *----------------------------------------------------------------------*/

static char char_from_level_array[] = {
  '$',				/* -1 == FRIBIDI_SENTINEL, indicating
				 * start or end of string. */
  /* 0-61 == 0-9,a-z,A-Z are the the only valid levels before resolving
   * implicits.  after that the level @ may be appear too. */
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
  'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
  'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D',
  'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
  'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
  'Y', 'Z',

  '@',				/* 62 == only must appear after resolving
				 * implicits. */

  '!',				/* 63 == FRIBIDI_LEVEL_INVALID, internal error,
				 * this level shouldn't be seen.  */

  '*', '*', '*', '*', '*'	/* >= 64 == overflows, this levels and higher
				 * levels show a real bug!. */
};

#define fribidi_char_from_level(level) char_from_level_array[(level) + 1]

static void
print_types_re (
  const FriBidiRun *pp
)
{
  fribidi_assert (pp);

  MSG ("  Run types  : ");
  for_run_list (pp, pp)
  {
    MSG5 ("%d:%d(%s)[%d] ",
	  pp->pos, pp->len, fribidi_get_bidi_type_name (pp->type), pp->level);
  }
  MSG ("\n");
}

static void
print_resolved_levels (
  const FriBidiRun *pp
)
{
  fribidi_assert (pp);

  MSG ("  Res. levels: ");
  for_run_list (pp, pp)
  {
    register FriBidiStrIndex i;
    for (i = RL_LEN (pp); i; i--)
      MSG2 ("%c", fribidi_char_from_level (RL_LEVEL (pp)));
  }
  MSG ("\n");
}

static void
print_resolved_types (
  const FriBidiRun *pp
)
{
  fribidi_assert (pp);

  MSG ("  Res. types : ");
  for_run_list (pp, pp)
  {
    FriBidiStrIndex i;
    for (i = RL_LEN (pp); i; i--)
      MSG2 ("%c", fribidi_char_from_bidi_type (pp->type));
  }
  MSG ("\n");
}

static void
print_bidi_string (
  /* input */
  const FriBidiCharType *bidi_types,
  const FriBidiStrIndex len
)
{
  register FriBidiStrIndex i;

  fribidi_assert (bidi_types);

  MSG ("  Org. types : ");
  for (i = 0; i < len; i++)
    MSG2 ("%c", fribidi_char_from_bidi_type (bidi_types[i]));
  MSG ("\n");
}
#endif /* DEBUG */


/*=========================================================================
 * define macros for push and pop the status in to / out of the stack
 *-------------------------------------------------------------------------*/

/* There are a few little points in pushing into and poping from the status
   stack:
   1. when the embedding level is not valid (more than
   FRIBIDI_BIDI_MAX_EXPLICIT_LEVEL=61), you must reject it, and not to push
   into the stack, but when you see a PDF, you must find the matching code,
   and if it was pushed in the stack, pop it, it means you must pop if and
   only if you have pushed the matching code, the over_pushed var counts the
   number of rejected codes so far.
   2. there's a more confusing point too, when the embedding level is exactly
   FRIBIDI_BIDI_MAX_EXPLICIT_LEVEL-1=60, an LRO or LRE is rejected
   because the new level would be FRIBIDI_BIDI_MAX_EXPLICIT_LEVEL+1=62, that
   is invalid; but an RLO or RLE is accepted because the new level is
   FRIBIDI_BIDI_MAX_EXPLICIT_LEVEL=61, that is valid, so the rejected codes
   may be not continuous in the logical order, in fact there are at most two
   continuous intervals of codes, with an RLO or RLE between them.  To support
   this case, the first_interval var counts the number of rejected codes in
   the first interval, when it is 0, means that there is only one interval.
*/

/* a. If this new level would be valid, then this embedding code is valid.
   Remember (push) the current embedding level and override status.
   Reset current level to this new level, and reset the override status to
   new_override.
   b. If the new level would not be valid, then this code is invalid. Don't
   change the current level or override status.
*/
#define PUSH_STATUS \
    FRIBIDI_BEGIN_STMT \
      if LIKELY(new_level <= FRIBIDI_BIDI_MAX_EXPLICIT_LEVEL) \
        { \
          if UNLIKELY(level == FRIBIDI_BIDI_MAX_EXPLICIT_LEVEL - 1) \
            first_interval = over_pushed; \
          status_stack[stack_size].level = level; \
          status_stack[stack_size].override = override; \
          stack_size++; \
          level = new_level; \
          override = new_override; \
        } else \
	  over_pushed++; \
    FRIBIDI_END_STMT

/* If there was a valid matching code, restore (pop) the last remembered
   (pushed) embedding level and directional override.
*/
#define POP_STATUS \
    FRIBIDI_BEGIN_STMT \
      if (stack_size) \
      { \
        if UNLIKELY(over_pushed > first_interval) \
          over_pushed--; \
        else \
          { \
            if LIKELY(over_pushed == first_interval) \
              first_interval = 0; \
            stack_size--; \
            level = status_stack[stack_size].level; \
            override = status_stack[stack_size].override; \
          } \
      } \
    FRIBIDI_END_STMT


/* Return the type of previous run or the SOR, if already at the start of
   a level run. */
#define PREV_TYPE_OR_SOR(pp) \
    ( \
      RL_LEVEL(pp->prev) == RL_LEVEL(pp) ? \
        RL_TYPE(pp->prev) : \
        FRIBIDI_LEVEL_TO_DIR(MAX(RL_LEVEL(pp->prev), RL_LEVEL(pp))) \
    )

/* Return the type of next run or the EOR, if already at the end of
   a level run. */
#define NEXT_TYPE_OR_EOR(pp) \
    ( \
      RL_LEVEL(pp->next) == RL_LEVEL(pp) ? \
        RL_TYPE(pp->next) : \
        FRIBIDI_LEVEL_TO_DIR(MAX(RL_LEVEL(pp->next), RL_LEVEL(pp))) \
    )


/* Return the embedding direction of a link. */
#define FRIBIDI_EMBEDDING_DIRECTION(link) \
    FRIBIDI_LEVEL_TO_DIR(RL_LEVEL(link))


FRIBIDI_ENTRY FriBidiParType
fribidi_get_par_direction (
  /* input */
  const FriBidiCharType *bidi_types,
  const FriBidiStrIndex len
)
{
  register FriBidiStrIndex i;

  fribidi_assert (bidi_types);

  for (i = 0; i < len; i++)
    if (FRIBIDI_IS_LETTER (bidi_types[i]))
      return FRIBIDI_IS_RTL (bidi_types[i]) ? FRIBIDI_PAR_RTL :
	FRIBIDI_PAR_LTR;

  return FRIBIDI_PAR_ON;
}

FRIBIDI_ENTRY FriBidiLevel
fribidi_get_par_embedding_levels (
  /* input */
  const FriBidiCharType *bidi_types,
  const FriBidiStrIndex len,
  /* input and output */
  FriBidiParType *pbase_dir,
  /* output */
  FriBidiLevel *embedding_levels
)
{
  FriBidiLevel base_level, max_level = 0;
  FriBidiParType base_dir;
  FriBidiRun *main_run_list = NULL, *explicits_list = NULL, *pp;
  fribidi_boolean status = false;

  if UNLIKELY
    (!len)
    {
      status = true;
      goto out;
    }

  DBG ("in fribidi_get_par_embedding_levels");

  fribidi_assert (bidi_types);
  fribidi_assert (pbase_dir);
  fribidi_assert (embedding_levels);

  /* Determinate character types */
  {
    /* Get run-length encoded character types */
    main_run_list = run_list_encode_bidi_types (bidi_types, len);
    if UNLIKELY
      (!main_run_list) goto out;
  }

  /* Find base level */
  /* If no strong base_dir was found, resort to the weak direction
     that was passed on input. */
  base_level = FRIBIDI_DIR_TO_LEVEL (*pbase_dir);
  if (!FRIBIDI_IS_STRONG (*pbase_dir))
    /* P2. P3. Search for first strong character and use its direction as
       base direction */
    {
      for_run_list (pp, main_run_list) if (FRIBIDI_IS_LETTER (RL_TYPE (pp)))
	{
	  base_level = FRIBIDI_DIR_TO_LEVEL (RL_TYPE (pp));
	  *pbase_dir = FRIBIDI_LEVEL_TO_DIR (base_level);
	  break;
	}
    }
  base_dir = FRIBIDI_LEVEL_TO_DIR (base_level);
  DBG2 ("  base level : %c", fribidi_char_from_level (base_level));
  DBG2 ("  base dir   : %c", fribidi_char_from_bidi_type (base_dir));

# if DEBUG
  if UNLIKELY
    (fribidi_debug_status ())
    {
      print_types_re (main_run_list);
    }
# endif	/* DEBUG */

  /* Explicit Levels and Directions */
  DBG ("explicit levels and directions");
  {
    FriBidiLevel level, new_level;
    FriBidiCharType override, new_override;
    FriBidiStrIndex i;
    int stack_size, over_pushed, first_interval;
    struct
    {
      FriBidiCharType override;	/* only LTR, RTL and ON are valid */
      FriBidiLevel level;
    } *status_stack;
    FriBidiRun temp_link;

/* explicits_list is a list like main_run_list, that holds the explicit
   codes that are removed from main_run_list, to reinsert them later by
   calling the shadow_run_list.
*/
    explicits_list = new_run_list ();
    if UNLIKELY
      (!explicits_list) goto out;

    /* X1. Begin by setting the current embedding level to the paragraph
       embedding level. Set the directional override status to neutral.
       Process each character iteratively, applying rules X2 through X9.
       Only embedding levels from 0 to 61 are valid in this phase. */

    level = base_level;
    override = FRIBIDI_TYPE_ON;
    /* stack */
    stack_size = 0;
    over_pushed = 0;
    first_interval = 0;
    status_stack = fribidi_malloc (sizeof (status_stack[0]) *
				   FRIBIDI_BIDI_MAX_RESOLVED_LEVELS);

    for_run_list (pp, main_run_list)
    {
      FriBidiCharType this_type = RL_TYPE (pp);
      if (FRIBIDI_IS_EXPLICIT_OR_BN (this_type))
	{
	  if (FRIBIDI_IS_STRONG (this_type))
	    {			/* LRE, RLE, LRO, RLO */
	      /* 1. Explicit Embeddings */
	      /*   X2. With each RLE, compute the least greater odd
	         embedding level. */
	      /*   X3. With each LRE, compute the least greater even
	         embedding level. */
	      /* 2. Explicit Overrides */
	      /*   X4. With each RLO, compute the least greater odd
	         embedding level. */
	      /*   X5. With each LRO, compute the least greater even
	         embedding level. */
	      new_override = FRIBIDI_EXPLICIT_TO_OVERRIDE_DIR (this_type);
	      for (i = RL_LEN (pp); i; i--)
		{
		  new_level =
		    ((level + FRIBIDI_DIR_TO_LEVEL (this_type) + 2) & ~1) -
		    FRIBIDI_DIR_TO_LEVEL (this_type);
		  PUSH_STATUS;
		}
	    }
	  else if (this_type == FRIBIDI_TYPE_PDF)
	    {
	      /* 3. Terminating Embeddings and overrides */
	      /*   X7. With each PDF, determine the matching embedding or
	         override code. */
	      for (i = RL_LEN (pp); i; i--)
		POP_STATUS;
	    }

	  /* X9. Remove all RLE, LRE, RLO, LRO, PDF, and BN codes. */
	  /* Remove element and add it to explicits_list */
	  RL_LEVEL (pp) = FRIBIDI_SENTINEL;
	  temp_link.next = pp->next;
	  move_node_before (pp, explicits_list);
	  pp = &temp_link;
	}
      else if (this_type == FRIBIDI_TYPE_BS)
	{
	  /* X8. All explicit directional embeddings and overrides are
	     completely terminated at the end of each paragraph. Paragraph
	     separators are not included in the embedding. */
	  break;
	}
      else
	{
	  /* X6. For all types besides RLE, LRE, RLO, LRO, and PDF:
	     a. Set the level of the current character to the current
	     embedding level.
	     b. Whenever the directional override status is not neutral,
	     reset the current character type to the directional override
	     status. */
	  RL_LEVEL (pp) = level;
	  if (!FRIBIDI_IS_NEUTRAL (override))
	    RL_TYPE (pp) = override;
	}
    }

    /* Implementing X8. It has no effect on a single paragraph! */
    level = base_level;
    override = FRIBIDI_TYPE_ON;
    stack_size = 0;
    over_pushed = 0;

    fribidi_free (status_stack);
  }
  /* X10. The remaining rules are applied to each run of characters at the
     same level. For each run, determine the start-of-level-run (sor) and
     end-of-level-run (eor) type, either L or R. This depends on the
     higher of the two levels on either side of the boundary (at the start
     or end of the paragraph, the level of the 'other' run is the base
     embedding level). If the higher level is odd, the type is R, otherwise
     it is L. */
  /* Resolving Implicit Levels can be done out of X10 loop, so only change
     of Resolving Weak Types and Resolving Neutral Types is needed. */

  compact_list (main_run_list);

# if DEBUG
  if UNLIKELY
    (fribidi_debug_status ())
    {
      print_types_re (main_run_list);
      print_bidi_string (bidi_types, len);
      print_resolved_levels (main_run_list);
      print_resolved_types (main_run_list);
    }
# endif	/* DEBUG */

  /* 4. Resolving weak types */
  DBG ("resolving weak types");
  {
    FriBidiCharType last_strong, prev_type_orig;
    fribidi_boolean w4;

    last_strong = base_dir;

    for_run_list (pp, main_run_list)
    {
      register FriBidiCharType prev_type, this_type, next_type;

      prev_type = PREV_TYPE_OR_SOR (pp);
      this_type = RL_TYPE (pp);
      next_type = NEXT_TYPE_OR_EOR (pp);

      if (FRIBIDI_IS_STRONG (prev_type))
	last_strong = prev_type;

      /* W1. NSM
         Examine each non-spacing mark (NSM) in the level run, and change the
         type of the NSM to the type of the previous character. If the NSM
         is at the start of the level run, it will get the type of sor. */
      /* Implementation note: it is important that if the previous character
         is not sor, then we should merge this run with the previous,
         because of rules like W5, that we assume all of a sequence of
         adjacent ETs are in one FriBidiRun. */
      if (this_type == FRIBIDI_TYPE_NSM)
	{
	  if (RL_LEVEL (pp->prev) == RL_LEVEL (pp))
	    pp = merge_with_prev (pp);
	  else
	    RL_TYPE (pp) = prev_type;
	  if (prev_type == next_type && RL_LEVEL (pp) == RL_LEVEL (pp->next))
	    {
	      pp = merge_with_prev (pp->next);
	    }
	  continue;		/* As we know the next condition cannot be true. */
	}

      /* W2: European numbers. */
      if (this_type == FRIBIDI_TYPE_EN && last_strong == FRIBIDI_TYPE_AL)
	{
	  RL_TYPE (pp) = FRIBIDI_TYPE_AN;

	  /* Resolving dependency of loops for rules W1 and W2, so we
	     can merge them in one loop. */
	  if (next_type == FRIBIDI_TYPE_NSM)
	    RL_TYPE (pp->next) = FRIBIDI_TYPE_AN;
	}
    }


    last_strong = base_dir;
    /* Resolving dependency of loops for rules W4 and W5, W5 may
       want to prevent W4 to take effect in the next turn, do this
       through "w4". */
    w4 = true;
    /* Resolving dependency of loops for rules W4 and W5 with W7,
       W7 may change an EN to L but it sets the prev_type_orig if needed,
       so W4 and W5 in next turn can still do their works. */
    prev_type_orig = FRIBIDI_TYPE_ON;

    for_run_list (pp, main_run_list)
    {
      register FriBidiCharType prev_type, this_type, next_type;

      prev_type = PREV_TYPE_OR_SOR (pp);
      this_type = RL_TYPE (pp);
      next_type = NEXT_TYPE_OR_EOR (pp);

      if (FRIBIDI_IS_STRONG (prev_type))
	last_strong = prev_type;

      /* W3: Change ALs to R. */
      if (this_type == FRIBIDI_TYPE_AL)
	{
	  RL_TYPE (pp) = FRIBIDI_TYPE_RTL;
	  w4 = true;
	  prev_type_orig = FRIBIDI_TYPE_ON;
	  continue;
	}

      /* W4. A single european separator changes to a european number.
         A single common separator between two numbers of the same type
         changes to that type. */
      if (w4
	  && RL_LEN (pp) == 1 && FRIBIDI_IS_ES_OR_CS (this_type)
	  && FRIBIDI_IS_NUMBER (prev_type_orig)
	  && prev_type_orig == next_type
	  && (prev_type_orig == FRIBIDI_TYPE_EN
	      || this_type == FRIBIDI_TYPE_CS))
	{
	  RL_TYPE (pp) = prev_type;
	  this_type = RL_TYPE (pp);
	}
      w4 = true;

      /* W5. A sequence of European terminators adjacent to European
         numbers changes to All European numbers. */
      if (this_type == FRIBIDI_TYPE_ET
	  && (prev_type_orig == FRIBIDI_TYPE_EN
	      || next_type == FRIBIDI_TYPE_EN))
	{
	  RL_TYPE (pp) = FRIBIDI_TYPE_EN;
	  w4 = false;
	  this_type = RL_TYPE (pp);
	}

      /* W6. Otherwise change separators and terminators to other neutral. */
      if (FRIBIDI_IS_NUMBER_SEPARATOR_OR_TERMINATOR (this_type))
	RL_TYPE (pp) = FRIBIDI_TYPE_ON;

      /* W7. Change european numbers to L. */
      if (this_type == FRIBIDI_TYPE_EN && last_strong == FRIBIDI_TYPE_LTR)
	{
	  RL_TYPE (pp) = FRIBIDI_TYPE_LTR;
	  prev_type_orig = (RL_LEVEL (pp) == RL_LEVEL (pp->next) ?
			    FRIBIDI_TYPE_EN : FRIBIDI_TYPE_ON);
	}
      else
	prev_type_orig = PREV_TYPE_OR_SOR (pp->next);
    }
  }

  compact_neutrals (main_run_list);

# if DEBUG
  if UNLIKELY
    (fribidi_debug_status ())
    {
      print_resolved_levels (main_run_list);
      print_resolved_types (main_run_list);
    }
# endif	/* DEBUG */

  /* 5. Resolving Neutral Types */
  DBG ("resolving neutral types");
  {
    /* N1. and N2.
       For each neutral, resolve it. */
    for_run_list (pp, main_run_list)
    {
      FriBidiCharType prev_type, this_type, next_type;

      /* "European and Arabic numbers are treated as though they were R"
         FRIBIDI_CHANGE_NUMBER_TO_RTL does this. */
      this_type = FRIBIDI_CHANGE_NUMBER_TO_RTL (RL_TYPE (pp));
      prev_type = FRIBIDI_CHANGE_NUMBER_TO_RTL (PREV_TYPE_OR_SOR (pp));
      next_type = FRIBIDI_CHANGE_NUMBER_TO_RTL (NEXT_TYPE_OR_EOR (pp));

      if (FRIBIDI_IS_NEUTRAL (this_type))
	RL_TYPE (pp) = (prev_type == next_type) ?
	  /* N1. */ prev_type :
	  /* N2. */ FRIBIDI_EMBEDDING_DIRECTION (pp);
    }
  }

  compact_list (main_run_list);

# if DEBUG
  if UNLIKELY
    (fribidi_debug_status ())
    {
      print_resolved_levels (main_run_list);
      print_resolved_types (main_run_list);
    }
# endif	/* DEBUG */

  /* 6. Resolving implicit levels */
  DBG ("resolving implicit levels");
  {
    max_level = base_level;

    for_run_list (pp, main_run_list)
    {
      FriBidiCharType this_type;
      int level;

      this_type = RL_TYPE (pp);
      level = RL_LEVEL (pp);

      /* I1. Even */
      /* I2. Odd */
      if (FRIBIDI_IS_NUMBER (this_type))
	RL_LEVEL (pp) = (level + 2) & ~1;
      else
	RL_LEVEL (pp) =
	  level +
	  (FRIBIDI_LEVEL_IS_RTL (level) ^ FRIBIDI_DIR_TO_LEVEL (this_type));

      if (RL_LEVEL (pp) > max_level)
	max_level = RL_LEVEL (pp);
    }
  }

  compact_list (main_run_list);

# if DEBUG
  if UNLIKELY
    (fribidi_debug_status ())
    {
      print_bidi_string (bidi_types, len);
      print_resolved_levels (main_run_list);
      print_resolved_types (main_run_list);
    }
# endif	/* DEBUG */

/* Reinsert the explicit codes & BN's that are already removed, from the
   explicits_list to main_run_list. */
  DBG ("reinserting explicit codes");
  if UNLIKELY
    (explicits_list->next != explicits_list)
    {
      register FriBidiRun *p;
      register fribidi_boolean stat =
	shadow_run_list (main_run_list, explicits_list, true);
      explicits_list = NULL;
      if UNLIKELY
	(!stat) goto out;

      /* Set level of inserted explicit chars to that of their previous
       * char, such that they do not affect reordering. */
      p = main_run_list->next;
      if (p != main_run_list && p->level == FRIBIDI_SENTINEL)
	p->level = base_level;
      for_run_list (p, main_run_list) if (p->level == FRIBIDI_SENTINEL)
	p->level = p->prev->level;
    }

# if DEBUG
  if UNLIKELY
    (fribidi_debug_status ())
    {
      print_types_re (main_run_list);
      print_resolved_levels (main_run_list);
      print_resolved_types (main_run_list);
    }
# endif	/* DEBUG */

  DBG ("reset the embedding levels, 1, 2, 3.");
  {
    register int j, state, pos;
    register FriBidiCharType char_type;
    register FriBidiRun *p, *q, *list;

    /* L1. Reset the embedding levels of some chars:
       1. segment separators,
       2. paragraph separators,
       3. any sequence of whitespace characters preceding a segment
       separator or paragraph separator, and
       ... (to be continued in fribidi_reorder_line()). */
    list = new_run_list ();
    if UNLIKELY
      (!list) goto out;
    q = list;
    state = 1;
    pos = len - 1;
    for (j = len - 1; j >= -1; j--)
      {
	/* close up the open link at the end */
	if (j >= 0)
	  char_type = bidi_types[j];
	else
	  char_type = FRIBIDI_TYPE_ON;
	if (!state && FRIBIDI_IS_SEPARATOR (char_type))
	  {
	    state = 1;
	    pos = j;
	  }
	else if (state && !FRIBIDI_IS_EXPLICIT_OR_SEPARATOR_OR_BN_OR_WS
		 (char_type))
	  {
	    state = 0;
	    p = new_run ();
	    if UNLIKELY
	      (!p)
	      {
		free_run_list (list);
		goto out;
	      }
	    p->pos = j + 1;
	    p->len = pos - j;
	    p->type = base_dir;
	    p->level = base_level;
	    move_node_before (p, q);
	    q = p;
	  }
      }
    if UNLIKELY
      (!shadow_run_list (main_run_list, list, false)) goto out;
  }

# if DEBUG
  if UNLIKELY
    (fribidi_debug_status ())
    {
      print_types_re (main_run_list);
      print_resolved_levels (main_run_list);
      print_resolved_types (main_run_list);
    }
# endif	/* DEBUG */

  {
    FriBidiStrIndex pos = 0;
    for_run_list (pp, main_run_list)
    {
      register FriBidiStrIndex l;
      register FriBidiLevel level = pp->level;
      for (l = pp->len; l; l--)
	embedding_levels[pos++] = level;
    }
  }

  status = true;

out:
  DBG ("leaving fribidi_get_par_embedding_levels");

  if (main_run_list)
    free_run_list (main_run_list);
  if UNLIKELY
    (explicits_list) free_run_list (explicits_list);

  return status ? max_level + 1 : 0;
}


static void
bidi_string_reverse (
  FriBidiChar *str,
  const FriBidiStrIndex len
)
{
  FriBidiStrIndex i;

  fribidi_assert (str);

  for (i = 0; i < len / 2; i++)
    {
      FriBidiChar tmp = str[i];
      str[i] = str[len - 1 - i];
      str[len - 1 - i] = tmp;
    }
}

static void
index_array_reverse (
  FriBidiStrIndex *arr,
  const FriBidiStrIndex len
)
{
  FriBidiStrIndex i;

  fribidi_assert (arr);

  for (i = 0; i < len / 2; i++)
    {
      FriBidiStrIndex tmp = arr[i];
      arr[i] = arr[len - 1 - i];
      arr[len - 1 - i] = tmp;
    }
}


FRIBIDI_ENTRY FriBidiLevel
fribidi_reorder_line (
  /* input */
  FriBidiFlags flags, /* reorder flags */
  const FriBidiCharType *bidi_types,
  const FriBidiStrIndex len,
  const FriBidiStrIndex off,
  const FriBidiParType base_dir,
  /* input and output */
  FriBidiLevel *embedding_levels,
  FriBidiChar *visual_str,
  /* output */
  FriBidiStrIndex *map
)
{
  fribidi_boolean status = false;
  FriBidiLevel max_level = 0;

  if UNLIKELY
    (len == 0)
    {
      status = true;
      goto out;
    }

  DBG ("in fribidi_reorder_line");

  fribidi_assert (bidi_types);
  fribidi_assert (embedding_levels);

  DBG ("reset the embedding levels, 4. whitespace at the end of line");
  {
    register FriBidiStrIndex i;

    /* L1. Reset the embedding levels of some chars:
       4. any sequence of white space characters at the end of the line. */
    for (i = off + len - 1; i >= off &&
	 FRIBIDI_IS_EXPLICIT_OR_BN_OR_WS (bidi_types[i]); i--)
      embedding_levels[i] = FRIBIDI_DIR_TO_LEVEL (base_dir);
  }

  /* 7. Reordering resolved levels */
  {
    register FriBidiLevel level;
    register FriBidiStrIndex i;

    /* Reorder both the outstring and the order array */
    {
      if (FRIBIDI_TEST_BITS (flags, FRIBIDI_FLAG_REORDER_NSM))
	{
	  /* L3. Reorder NSMs. */
	  for (i = off + len - 1; i >= off; i--)
	    if (FRIBIDI_LEVEL_IS_RTL (embedding_levels[i])
		&& bidi_types[i] == FRIBIDI_TYPE_NSM)
	      {
		register FriBidiStrIndex seq_end = i;
		level = embedding_levels[i];

		for (i--; i >= off &&
		     FRIBIDI_IS_EXPLICIT_OR_BN_OR_NSM (bidi_types[i])
		     && embedding_levels[i] == level; i--)
		  ;

		if (i < off || embedding_levels[i] != level)
		  {
		    i++;
		    DBG ("warning: NSM(s) at the beggining of level run");
		  }

		if (visual_str)
		  {
		    bidi_string_reverse (visual_str + i, seq_end - i + 1);
		  }
		if (map)
		  {
		    index_array_reverse (map + i, seq_end - i + 1);
		  }
	      }
	}

      /* Find max_level of the line.  We don't reuse the paragraph
       * max_level, both for a cleaner API, and that the line max_level
       * may be far less than paragraph max_level. */
      for (i = off + len - 1; i >= off; i--)
	if (embedding_levels[i] > max_level)
	  max_level = embedding_levels[i];

      /* L2. Reorder. */
      for (level = max_level; level > 0; level--)
	for (i = off + len - 1; i >= off; i--)
	  if (embedding_levels[i] >= level)
	    {
	      /* Find all stretches that are >= level_idx */
	      register FriBidiStrIndex seq_end = i;
	      for (i--; i >= off && embedding_levels[i] >= level; i--)
		;

	      if (visual_str)
		bidi_string_reverse (visual_str + i + 1, seq_end - i);
	      if (map)
		index_array_reverse (map + i + 1, seq_end - i);
	    }
    }

  }

  status = true;

out:

  return status ? max_level + 1 : 0;
}

/* Editor directions:
 * vim:textwidth=78:tabstop=8:shiftwidth=2:autoindent:cindent
 */
