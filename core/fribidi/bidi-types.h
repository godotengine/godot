/* FriBidi
 * bidi-types.h - define internal bidi types
 *
 * $Id: bidi-types.h,v 1.7 2006-01-31 03:23:12 behdad Exp $
 * $Author: behdad $
 * $Date: 2006-01-31 03:23:12 $
 * $Revision: 1.7 $
 * $Source: /home/behdad/src/fdo/fribidi/togit/git/../fribidi/fribidi2/lib/bidi-types.h,v $
 *
 * Author:
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
#ifndef _BIDI_TYPES_H
#define _BIDI_TYPES_H

#include "common.h"

#include <fribidi-types.h>
#include <fribidi-bidi-types.h>

#include <fribidi-begindecls.h>

#define FRIBIDI_LEVEL_INVALID FRIBIDI_BIDI_MAX_RESOLVED_LEVELS
#define FRIBIDI_SENTINEL -1

#if DEBUG+0

#define fribidi_char_from_bidi_type FRIBIDI_PRIVATESPACE(char_from_bidi_type)
char
fribidi_char_from_bidi_type (
  FriBidiCharType t		/* input bidi type */
) FRIBIDI_GNUC_HIDDEN;

#endif /* DEBUG */

#include <fribidi-enddecls.h>

#endif /* !_BIDI_TYPES_H */
/* Editor directions:
 * vim:textwidth=78:tabstop=8:shiftwidth=2:autoindent:cindent
 */
