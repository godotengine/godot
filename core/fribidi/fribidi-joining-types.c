/* FriBidi
 * fribidi-joining-types.c - character joining types
 *
 * $Id: fribidi-joining-types.c,v 1.5 2006-01-31 03:23:13 behdad Exp $
 * $Author: behdad $
 * $Date: 2006-01-31 03:23:13 $
 * $Revision: 1.5 $
 * $Source: /home/behdad/src/fdo/fribidi/togit/git/../fribidi/fribidi2/lib/fribidi-joining-types.c,v $
 *
 * Authors:
 *   Behdad Esfahbod, 2001, 2002, 2004
 *
 * Copyright (C) 2004 Sharif FarsiWeb, Inc.
 * Copyright (C) 2001,2002 Behdad Esfahbod
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

#include <fribidi-joining-types.h>

#include "joining-types.h"

enum FriBidiJoiningTypeShortEnum
{
# define _FRIBIDI_ADD_TYPE(TYPE,SYMBOL) TYPE = FRIBIDI_JOINING_TYPE_##TYPE,
# include "fribidi-joining-types-list.h"
# undef _FRIBIDI_ADD_TYPE
  _FRIBIDI_NUM_TYPES
};

#include "joining-type.tab.i"

FRIBIDI_ENTRY FriBidiJoiningType
fribidi_get_joining_type (
  /* input */
  FriBidiChar ch
)
{
  return FRIBIDI_GET_JOINING_TYPE (ch);
}

FRIBIDI_ENTRY void
fribidi_get_joining_types (
  /* input */
  const FriBidiChar *str,
  const FriBidiStrIndex len,
  /* output */
  FriBidiJoiningType *jtypes
)
{
  register FriBidiStrIndex i = len;
  for (; i; i--)
    {
      *jtypes++ = FRIBIDI_GET_JOINING_TYPE (*str);
      str++;
    }
}

FRIBIDI_ENTRY const char *
fribidi_get_joining_type_name (
  /* input */
  FriBidiJoiningType j
)
{
  switch (j)
    {
#   define _FRIBIDI_ADD_TYPE(TYPE,SYMBOL) case FRIBIDI_JOINING_TYPE_##TYPE: return STRINGIZE(TYPE);
#   include "fribidi-joining-types-list.h"
#   undef _FRIBIDI_ADD_TYPE
    default:
      return "?";
    }
}

#if DEBUG+0

char
fribidi_char_from_joining_type (
  /* input */
  FriBidiJoiningType j,
  fribidi_boolean visual
)
{
  /* switch left and right if on visual run */
  if (visual & ((FRIBIDI_JOINS_RIGHT (j) && !FRIBIDI_JOINS_LEFT (j)) |
		(!FRIBIDI_JOINS_RIGHT (j) && FRIBIDI_JOINS_LEFT (j))))
    j ^= FRIBIDI_MASK_JOINS_RIGHT | FRIBIDI_MASK_JOINS_LEFT;

#   define _FRIBIDI_ADD_TYPE(TYPE,SYMBOL)	\
	if (FRIBIDI_IS_JOINING_TYPE_##TYPE(j)) return SYMBOL;
#   include "fribidi-joining-types-list.h"
#   undef _FRIBIDI_ADD_TYPE

  return '?';
}

#endif /* DEBUG */

/* Editor directions:
 * vim:textwidth=78:tabstop=8:shiftwidth=2:autoindent:cindent
 */
