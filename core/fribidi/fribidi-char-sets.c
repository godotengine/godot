/* FriBidi
 * fribidi-char-sets.c - character set conversion routines
 *
 * $Id: fribidi-char-sets.c,v 1.7 2006-01-31 03:23:12 behdad Exp $
 * $Author: behdad $
 * $Date: 2006-01-31 03:23:12 $
 * $Revision: 1.7 $
 * $Source: /home/behdad/src/fdo/fribidi/togit/git/../fribidi/fribidi2/charset/fribidi-char-sets.c,v $
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
#include "fribidi-char-sets.h"

#include "fribidi-char-sets-cap-rtl.h"
#include "fribidi-char-sets-utf8.h"
#include "fribidi-char-sets-iso8859-6.h"
#include "fribidi-char-sets-cp1256.h"
#include "fribidi-char-sets-iso8859-8.h"
#include "fribidi-char-sets-cp1255.h"

typedef struct
{
  FriBidiChar (
  *charset_to_unicode_c
  ) (
  char ch
  );

  FriBidiStrIndex (
  *charset_to_unicode
  ) (
  const char *s,
  FriBidiStrIndex len,
  FriBidiChar *us
  );

  char (
  *unicode_to_charset_c
  ) (
  FriBidiChar uch
  );

  FriBidiStrIndex (
  *unicode_to_charset
  ) (
  const FriBidiChar *us,
  FriBidiStrIndex len,
  char *s
  );

  const char *name;

  const char *title;

  const char *(
  *desc
  ) (
  void
  );
}
FriBidiCharSetHandler;

static FriBidiCharSetHandler char_sets[FRIBIDI_CHAR_SETS_NUM + 1] = {
  {NULL, NULL, NULL, NULL, "N/A", "Character set not available", NULL},
# define _FRIBIDI_ADD_CHAR_SET_ONE2ONE(CHAR_SET, char_set) \
  { \
    fribidi_##char_set##_to_unicode_c, \
    NULL, \
    fribidi_unicode_to_##char_set##_c, \
    NULL, \
    fribidi_char_set_name_##char_set, \
    fribidi_char_set_title_##char_set, \
    fribidi_char_set_desc_##char_set \
  },
# define _FRIBIDI_ADD_CHAR_SET_OTHERS(CHAR_SET, char_set) \
  { \
    NULL, \
    fribidi_##char_set##_to_unicode, \
    NULL, \
    fribidi_unicode_to_##char_set, \
    fribidi_char_set_name_##char_set, \
    fribidi_char_set_title_##char_set, \
    fribidi_char_set_desc_##char_set \
  },
# include "fribidi-char-sets-list.h"
# undef _FRIBIDI_ADD_CHAR_SET_OTHERS
# undef _FRIBIDI_ADD_CHAR_SET_ONE2ONE
};

#if FRIBIDI_USE_GLIB+0
# include <glib/gstrfuncs.h>
# define fribidi_strcasecmp g_ascii_strcasecmp
#else /* !FRIBIDI_USE_GLIB */
static char
toupper (
  /* input */
  char c
)
{
  return c < 'a' || c > 'z' ? c : c + 'A' - 'a';
}

static int
fribidi_strcasecmp (
  /* input */
  const char *s1,
  const char *s2
)
{
  while (*s1 && toupper (*s1) == toupper (*s2))
    {
      s1++;
      s2++;
    }
  return toupper (*s1) - toupper (*s2);
}
#endif /* !FRIBIDI_USE_GLIB */

FRIBIDI_ENTRY FriBidiCharSet
fribidi_parse_charset (
  /* input */
  const char *s
)
{
  int i;

  for (i = FRIBIDI_CHAR_SETS_NUM; i; i--)
    if (fribidi_strcasecmp (s, char_sets[i].name) == 0)
      return i;

  return FRIBIDI_CHAR_SET_NOT_FOUND;
}

FRIBIDI_ENTRY FriBidiStrIndex
fribidi_charset_to_unicode (
  /* input */
  FriBidiCharSet char_set,
  const char *s,
  FriBidiStrIndex len,
  /* output */
  FriBidiChar *us
)
{

  if (char_sets[char_set].charset_to_unicode)
    return (*char_sets[char_set].charset_to_unicode) (s, len, us);
  else if (char_sets[char_set].charset_to_unicode_c)
    {
      register FriBidiStrIndex l;
      for (l = len; l; l--)
	*us++ = (*char_sets[char_set].charset_to_unicode_c) (*s++);
      return len;
    }
  else
    return 0;
}

FRIBIDI_ENTRY FriBidiStrIndex
fribidi_unicode_to_charset (
  /* input */
  FriBidiCharSet char_set,
  const FriBidiChar *us,
  FriBidiStrIndex len,
  /* output */
  char *s
)
{
  if (char_sets[char_set].unicode_to_charset)
    return (*char_sets[char_set].unicode_to_charset) (us, len, s);
  else if (char_sets[char_set].unicode_to_charset_c)
    {
      register FriBidiStrIndex l;
      for (l = len; l; l--)
	*s++ = (*char_sets[char_set].unicode_to_charset_c) (*us++);
      *s = '\0';
      return len;
    }
  else
    return 0;
}

FRIBIDI_ENTRY const char *
fribidi_char_set_name (
  /* input */
  FriBidiCharSet char_set
)
{
  return char_sets[char_set].name ? char_sets[char_set].name : "";
}

FRIBIDI_ENTRY const char *
fribidi_char_set_title (
  /* input */
  FriBidiCharSet char_set
)
{
  return char_sets[char_set].title ? char_sets[char_set].
    title : fribidi_char_set_name (char_set);
}


FRIBIDI_ENTRY const char *
fribidi_char_set_desc (
  /* input */
  FriBidiCharSet char_set
)
{
  return char_sets[char_set].desc ? char_sets[char_set].desc () : NULL;
}

/* Editor directions:
 * vim:textwidth=78:tabstop=8:shiftwidth=2:autoindent:cindent
 */
