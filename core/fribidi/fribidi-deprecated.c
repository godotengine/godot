/* FriBidi
 * fribidi-deprecated.c - deprecated interfaces.
 *
 * $Id: fribidi-deprecated.c,v 1.6 2006-06-01 22:53:55 behdad Exp $
 * $Author: behdad $
 * $Date: 2006-06-01 22:53:55 $
 * $Revision: 1.6 $
 * $Source: /home/behdad/src/fdo/fribidi/togit/git/../fribidi/fribidi2/lib/fribidi-deprecated.c,v $
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

#undef FRIBIDI_NO_DEPRECATED

#include <fribidi-deprecated.h>
#include <fribidi.h>

#if FRIBIDI_NO_DEPRECATED+0
#else

static FriBidiFlags flags = FRIBIDI_FLAGS_DEFAULT | FRIBIDI_FLAGS_ARABIC;

FRIBIDI_ENTRY fribidi_boolean
fribidi_set_mirroring (
  /* input */
  fribidi_boolean state
)
{
  return FRIBIDI_ADJUST_AND_TEST_BITS (flags, FRIBIDI_FLAG_SHAPE_MIRRORING, state);
}

FRIBIDI_ENTRY fribidi_boolean
fribidi_mirroring_status (
  void
)
{
  return FRIBIDI_TEST_BITS (flags, FRIBIDI_FLAG_SHAPE_MIRRORING);
}

FRIBIDI_ENTRY fribidi_boolean
fribidi_set_reorder_nsm (
  /* input */
  fribidi_boolean state
)
{
  return FRIBIDI_ADJUST_AND_TEST_BITS (flags, FRIBIDI_FLAG_REORDER_NSM, state);
}

fribidi_boolean
fribidi_reorder_nsm_status (
  void
)
{
  return FRIBIDI_TEST_BITS (flags, FRIBIDI_FLAG_REORDER_NSM);
}




FRIBIDI_ENTRY FriBidiLevel
fribidi_log2vis_get_embedding_levels (
  const FriBidiCharType *bidi_types,	/* input list of bidi types as returned by
					   fribidi_get_bidi_types() */
  const FriBidiStrIndex len,	/* input string length of the paragraph */
  FriBidiParType *pbase_dir,	/* requested and resolved paragraph
				 * base direction */
  FriBidiLevel *embedding_levels	/* output list of embedding levels */
)
{
  return fribidi_get_par_embedding_levels (bidi_types, len, pbase_dir, embedding_levels);
}

FRIBIDI_ENTRY FriBidiCharType
fribidi_get_type (
  FriBidiChar ch		/* input character */
)
{
  return fribidi_get_bidi_type (ch);
}

FRIBIDI_ENTRY FriBidiCharType
fribidi_get_type_internal (
  FriBidiChar ch		/* input character */
)
{
  return fribidi_get_bidi_type (ch);
}



FRIBIDI_ENTRY FriBidiStrIndex
fribidi_remove_bidi_marks (
  FriBidiChar *str,
  const FriBidiStrIndex len,
  FriBidiStrIndex *positions_to_this,
  FriBidiStrIndex *position_from_this_list,
  FriBidiLevel *embedding_levels
)
{
  register FriBidiStrIndex i, j = 0;
  fribidi_boolean private_from_this = false;
  fribidi_boolean status = false;

  if UNLIKELY
    (len == 0)
    {
      status = true;
      goto out;
    }

  DBG ("in fribidi_remove_bidi_marks");

  fribidi_assert (str);

  /* If to_this is not NULL, we must have from_this as well. If it is
     not given by the caller, we have to make a private instance of it. */
  if (positions_to_this && !position_from_this_list)
    {
      position_from_this_list = fribidi_malloc (sizeof
						(position_from_this_list[0]) *
						len);
      if UNLIKELY
	(!position_from_this_list) goto out;
      private_from_this = true;
      for (i = 0; i < len; i++)
	position_from_this_list[positions_to_this[i]] = i;
    }

  for (i = 0; i < len; i++)
    if (!FRIBIDI_IS_EXPLICIT_OR_BN (fribidi_get_bidi_type (str[i]))
	&& str[i] != FRIBIDI_CHAR_LRM && str[i] != FRIBIDI_CHAR_RLM)
      {
	str[j] = str[i];
	if (embedding_levels)
	  embedding_levels[j] = embedding_levels[i];
	if (position_from_this_list)
	  position_from_this_list[j] = position_from_this_list[i];
	j++;
      }

  /* Convert the from_this list to to_this */
  if (positions_to_this)
    {
      for (i = 0; i < len; i++)
	positions_to_this[i] = -1;
      for (i = 0; i < len; i++)
	positions_to_this[position_from_this_list[i]] = i;
    }

  status = true;

out:

  if (private_from_this)
    fribidi_free (position_from_this_list);

  return status ? j : -1;
}



FRIBIDI_ENTRY FriBidiLevel
fribidi_log2vis (
  /* input */
  const FriBidiChar *str,
  FriBidiStrIndex len,
  /* input and output */
  FriBidiParType *pbase_dir,
  /* output */
  FriBidiChar *visual_str,
  FriBidiStrIndex *positions_L_to_V,
  FriBidiStrIndex *positions_V_to_L,
  FriBidiLevel *embedding_levels
)
{
  register FriBidiStrIndex i;
  FriBidiLevel max_level = 0;
  fribidi_boolean private_V_to_L = false;
  fribidi_boolean private_embedding_levels = false;
  fribidi_boolean status = false;
  FriBidiArabicProp *ar_props = NULL;
  FriBidiCharType *bidi_types = NULL;

  if UNLIKELY
    (len == 0)
    {
      status = true;
      goto out;
    }

  DBG ("in fribidi_log2vis");

  fribidi_assert (str);
  fribidi_assert (pbase_dir);

  bidi_types = fribidi_malloc (len * sizeof bidi_types[0]);
  if (!bidi_types)
    goto out;

  fribidi_get_bidi_types (str, len, bidi_types);

  if (!embedding_levels)
    {
      embedding_levels = fribidi_malloc (len * sizeof embedding_levels[0]);
      if (!embedding_levels)
	goto out;
      private_embedding_levels = true;
    }

  max_level = fribidi_get_par_embedding_levels (bidi_types, len, pbase_dir,
						embedding_levels) - 1;
  if UNLIKELY
    (max_level < 0) goto out;

  /* If l2v is to be calculated we must have v2l as well. If it is not
     given by the caller, we have to make a private instance of it. */
  if (positions_L_to_V && !positions_V_to_L)
    {
      positions_V_to_L =
	(FriBidiStrIndex *) fribidi_malloc (sizeof (FriBidiStrIndex) * len);
      if (!positions_V_to_L)
	goto out;
      private_V_to_L = true;
    }

  /* Set up the ordering array to identity order */
  if (positions_V_to_L)
    {
      for (i = 0; i < len; i++)
	positions_V_to_L[i] = i;
    }


  if (visual_str)
    {
      /* Using memcpy instead
      for (i = len - 1; i >= 0; i--)
	visual_str[i] = str[i];
      */
      memcpy (visual_str, str, len * sizeof (*visual_str));

      /* Arabic joining */
      ar_props = fribidi_malloc (len * sizeof ar_props[0]);
      fribidi_get_joining_types (str, len, ar_props);
      fribidi_join_arabic (bidi_types, len, embedding_levels, ar_props);

      fribidi_shape (flags, embedding_levels, len, ar_props, visual_str);
    }

  /* line breaking goes here, but we assume one line in this function */

  /* and this should be called once per line, but again, we assume one
   * line in this deprecated function */
  status =
    fribidi_reorder_line (flags, bidi_types, len, 0, *pbase_dir,
			  embedding_levels, visual_str,
			  positions_V_to_L);

  /* Convert the v2l list to l2v */
  if (positions_L_to_V)
    {
      for (i = 0; i < len; i++)
	positions_L_to_V[i] = -1;
      for (i = 0; i < len; i++)
	positions_L_to_V[positions_V_to_L[i]] = i;
    }

out:

  if (private_V_to_L)
    fribidi_free (positions_V_to_L);

  if (private_embedding_levels)
    fribidi_free (embedding_levels);

  if (ar_props)
    fribidi_free (ar_props);

  if (bidi_types)
    fribidi_free (bidi_types);

  return status ? max_level + 1 : 0;
}

#endif /* !FRIBIDI_NO_DEPRECATED */

/* Editor directions:
 * vim:textwidth=78:tabstop=8:shiftwidth=2:autoindent:cindent
 */
