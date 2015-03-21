/* FriBidi
 * fribidi.c - Unicode bidirectional and Arabic joining/shaping algorithms
 *
 * $Id: fribidi.c,v 1.18 2006-01-31 03:23:13 behdad Exp $
 * $Author: behdad $
 * $Date: 2006-01-31 03:23:13 $
 * $Revision: 1.18 $
 * $Source: /home/behdad/src/fdo/fribidi/togit/git/../fribidi/fribidi2/lib/fribidi.c,v $
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

#include <fribidi.h>

#if DEBUG+0
static int flag_debug = false;
#endif

FRIBIDI_ENTRY fribidi_boolean
fribidi_debug_status (
  void
)
{
#if DEBUG+0
  return flag_debug;
#else
  return false;
#endif
}

FRIBIDI_ENTRY fribidi_boolean
fribidi_set_debug (
  /* input */
  fribidi_boolean state
)
{
#if DEBUG+0
  return flag_debug = state;
#else
  return false;
#endif
}



const char *fribidi_unicode_version = FRIBIDI_UNICODE_VERSION;

const char *fribidi_version_info =
  "(" FRIBIDI_NAME ") " FRIBIDI_VERSION "\n"
  "interface version " FRIBIDI_INTERFACE_VERSION_STRING ",\n"
  "Unicode Character Database version " FRIBIDI_UNICODE_VERSION ",\n"
  "Configure options"
#if DEBUG+0
  " --enable-debug"
#endif /* DEBUG */
#if USE_SIMPLE_MALLOC+0
  " --enable-malloc"
#endif /* USE_SIMPLE_MALLOC */
#if FRIBIDI_CHARSETS+0
#else
  " --disable-charsets"
#endif /* !FRIBIDI_CHARSETS */
#if FRIBIDI_USE_GLIB+0
  " --with-glib"
#else /* !FRIBIDI_USE_GLIB */
  " --without-glib"
#endif /* !FRIBIDI_USE_GLIB */
  ".\n\n"
  "Copyright (C) 2004  Sharif FarsiWeb, Inc.\n"
  "Copyright (C) 2001, 2002, 2004, 2005  Behdad Esfahbod\n"
  "Copyright (C) 1999, 2000  Dov Grobgeld\n"
  FRIBIDI_NAME " comes with NO WARRANTY, to the extent permitted by law.\n"
  "You may redistribute copies of " FRIBIDI_NAME " under\n"
  "the terms of the GNU Lesser General Public License.\n"
  "For more information about these matters, see the file named COPYING.\n\n"
  "Written by Behdad Esfahbod and Dov Grobgeld.\n";

/* Editor directions:
 * vim:textwidth=78:tabstop=8:shiftwidth=2:autoindent:cindent
 */
