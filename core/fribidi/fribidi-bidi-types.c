/* FriBidi
 * fribidi-bidi-types.c - character bidi types
 *
 * $Id: fribidi-bidi-types.c,v 1.9 2006-01-31 03:23:13 behdad Exp $
 * $Author: behdad $
 * $Date: 2006-01-31 03:23:13 $
 * $Revision: 1.9 $
 * $Source: /home/behdad/src/fdo/fribidi/togit/git/../fribidi/fribidi2/lib/fribidi-bidi-types.c,v $
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

#include <fribidi-bidi-types.h>

#include "bidi-types.h"

enum FriBidiCharTypeLinearEnum
{
# define _FRIBIDI_ADD_TYPE(TYPE,SYMBOL) TYPE,
# include "fribidi-bidi-types-list.h"
# undef _FRIBIDI_ADD_TYPE
  _FRIBIDI_NUM_TYPES
};

#include "bidi-type.tab.i"

/* Map FriBidiCharTypeLinearEnum to FriBidiCharType. */
static const FriBidiCharType linear_enum_to_char_type[] = {
# define _FRIBIDI_ADD_TYPE(TYPE,SYMBOL) FRIBIDI_TYPE_##TYPE,
# include "fribidi-bidi-types-list.h"
# undef _FRIBIDI_ADD_TYPE
};

FRIBIDI_ENTRY FriBidiCharType
fribidi_get_bidi_type (
  /* input */
  FriBidiChar ch
)
{
  return linear_enum_to_char_type[FRIBIDI_GET_BIDI_TYPE (ch)];
}

FRIBIDI_ENTRY void
fribidi_get_bidi_types (
  /* input */
  const FriBidiChar *str,
  const FriBidiStrIndex len,
  /* output */
  FriBidiCharType *btypes
)
{
  register FriBidiStrIndex i = len;
  for (; i; i--)
    {
      *btypes++ = linear_enum_to_char_type[FRIBIDI_GET_BIDI_TYPE (*str)];
      str++;
    }
}

FRIBIDI_ENTRY const char *
fribidi_get_bidi_type_name (
  /* input */
  FriBidiCharType t
)
{
  switch (t)
    {
#   define _FRIBIDI_ADD_TYPE(TYPE,SYMBOL) case FRIBIDI_TYPE_##TYPE: return STRINGIZE(TYPE);
#   define _FRIBIDI_ALL_TYPES
#   include "fribidi-bidi-types-list.h"
#   undef _FRIBIDI_ALL_TYPES
#   undef _FRIBIDI_ADD_TYPE
    default:
      return "?";
    }
}

#if DEBUG+0

char
fribidi_char_from_bidi_type (
  /* input */
  FriBidiCharType t
)
{
  switch (t)
    {
#   define _FRIBIDI_ADD_TYPE(TYPE,SYMBOL) case FRIBIDI_TYPE_##TYPE: return SYMBOL;
#   define _FRIBIDI_ALL_TYPES
#   include "fribidi-bidi-types-list.h"
#   undef _FRIBIDI_ALL_TYPES
#   undef _FRIBIDI_ADD_TYPE
    default:
      return '?';
    }
}

#endif /* DEBUG */

/* Editor directions:
 * vim:textwidth=78:tabstop=8:shiftwidth=2:autoindent:cindent
 */
