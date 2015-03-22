/* FriBidi
 * fribidi-types.h - define data types for the rest of the library
 *
 * $Id: fribidi-types.h,v 1.13 2010-02-24 19:40:04 behdad Exp $
 * $Author: behdad $
 * $Date: 2010-02-24 19:40:04 $
 * $Revision: 1.13 $
 * $Source: /home/behdad/src/fdo/fribidi/togit/git/../fribidi/fribidi2/lib/fribidi-types.h,v $
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
#ifndef _FRIBIDI_TYPES_H
#define _FRIBIDI_TYPES_H

#include "fribidi-common.h"

#include "fribidi-begindecls.h"


#if FRIBIDI_USE_GLIB+0
# ifndef __FRIBIDI_DOC
#  include <glib.h>
# endif	/* !__FRIBIDI_DOC */
# define FRIBIDI_INT8_LOCAL		gint8
# define FRIBIDI_INT16_LOCAL		gint16
# define FRIBIDI_INT32_LOCAL		gint32
# define FRIBIDI_UINT8_LOCAL		guint8
# define FRIBIDI_UINT16_LOCAL		guint16
# define FRIBIDI_UINT32_LOCAL		guint32
# define FRIBIDI_BOOLEAN_LOCAL		gboolean
# define FRIBIDI_UNICHAR_LOCAL		gunichar
#else /* !FRIBIDI_USE_GLIB */
# if defined(HAVE_INTTYPES_H) || defined(HAVE_STDINT_H)
#  ifndef __FRIBIDI_DOC
#   if HAVE_INTTYPES_H
#    include <inttypes.h>
#   elif HAVE_STDINT_H
#    include <stdint.h>
#   endif /* !HAVE_STDINT_H */
#  endif /* !__FRIBIDI_DOC */
#  define FRIBIDI_INT8_LOCAL		int8_t
#  define FRIBIDI_INT16_LOCAL		int16_t
#  define FRIBIDI_INT32_LOCAL		int32_t
#  define FRIBIDI_UINT8_LOCAL		uint8_t
#  define FRIBIDI_UINT16_LOCAL		uint16_t
#  define FRIBIDI_UINT32_LOCAL		uint32_t
# else /* no int types */
#  define FRIBIDI_INT8_LOCAL		signed char
#  define FRIBIDI_UINT8_LOCAL		unsigned char
#  if !defined(FRIBIDI_SIZEOF_INT) || FRIBIDI_SIZEOF_INT >= 4
#   define FRIBIDI_INT16_LOCAL		signed short
#   define FRIBIDI_UINT16_LOCAL		unsigned short
#   define FRIBIDI_INT32_LOCAL		signed int
#   define FRIBIDI_UINT32_LOCAL		unsigned int
#  else	/* SIZEOF_INT < 4 */
#   define FRIBIDI_INT16_LOCAL		signed int
#   define FRIBIDI_UINT16_LOCAL		unsigned int
#   define FRIBIDI_INT32_LOCAL		signed long
#   define FRIBIDI_UINT32_LOCAL		unsigned long
#  endif /* SIZEOF_INT < 4 */
# endif	/* no int types */
# define FRIBIDI_BOOLEAN_LOCAL		int
# if SIZEOF_WCHAR_T >= 4
#  ifndef __FRIBIDI_DOC
#   if STDC_HEADERS
#    include <stdlib.h>
#    include <stddef.h>
#   else /* !STDC_HEADERS */
#    if HAVE_STDLIB_H
#     include <stdlib.h>
#    endif /* !HAVE_STDLIB_H */
#   endif /* !STDC_HEADERS */
#  endif /* !__FRIBIDI_DOC */
#  define FRIBIDI_UNICHAR_LOCAL		wchar_t
# else /* SIZEOF_WCHAR_T < 4 */
#  define FRIBIDI_UNICHAR_LOCAL		fribidi_uint32
# endif	/* SIZEOF_WCHAR_T < 4 */
#endif /* !FRIBIDI_USE_GLIB */

#if FRIBIDI_INT_TYPES+0
#else
# define FRIBIDI_INT8	FRIBIDI_INT8_LOCAL
# define FRIBIDI_INT16	FRIBIDI_INT16_LOCAL
# define FRIBIDI_INT32	FRIBIDI_INT32_LOCAL
# define FRIBIDI_UINT8	FRIBIDI_UINT8_LOCAL
# define FRIBIDI_UINT16	FRIBIDI_UINT16_LOCAL
# define FRIBIDI_UINT32	FRIBIDI_UINT32_LOCAL
#endif /* !FRIBIDI_INT_TYPES */
#ifndef FRIBIDI_BOOLEAN
# define FRIBIDI_BOOLEAN	FRIBIDI_BOOLEAN_LOCAL
#endif /* !FRIBIDI_BOOLEAN */
#ifndef FRIBIDI_UNICHAR
# define FRIBIDI_UNICHAR FRIBIDI_UNICHAR_LOCAL
#endif /* !FRIBIDI_UNICHAR */
#ifndef FRIBIDI_STR_INDEX
# define FRIBIDI_STR_INDEX int
#endif /* FRIBIDI_STR_INDEX */


typedef FRIBIDI_UINT8 fribidi_int8;
typedef FRIBIDI_INT16 fribidi_int16;
typedef FRIBIDI_INT32 fribidi_int32;
typedef FRIBIDI_UINT8 fribidi_uint8;
typedef FRIBIDI_UINT16 fribidi_uint16;
typedef FRIBIDI_UINT32 fribidi_uint32;
typedef FRIBIDI_BOOLEAN fribidi_boolean;

typedef FRIBIDI_UNICHAR FriBidiChar;
typedef FRIBIDI_STR_INDEX FriBidiStrIndex;


#ifndef FRIBIDI_MAX_STRING_LENGTH
# define FRIBIDI_MAX_STRING_LENGTH (sizeof (FriBidiStrIndex) == 2 ?	\
		0x7FFF : (sizeof (FriBidiStrIndex) == 1 ? \
		0x7F : 0x7FFFFFFFL))
#endif

/* A few macros for working with bits */

#define FRIBIDI_TEST_BITS(x, mask) (((x) & (mask)) ? 1 : 0)

#define FRIBIDI_INCLUDE_BITS(x, mask) ((x) | (mask))

#define FRIBIDI_EXCLUDE_BITS(x, mask) ((x) & ~(mask))

#define FRIBIDI_SET_BITS(x, mask)	((x) |= (mask))

#define FRIBIDI_UNSET_BITS(x, mask)	((x) &= ~(mask))

#define FRIBIDI_ADJUST_BITS(x, mask, cond)	\
	((x) = ((x) & ~(mask)) | ((cond) ? (mask) : 0))

#define FRIBIDI_ADJUST_AND_TEST_BITS(x, mask, cond)	\
	FRIBIDI_TEST_BITS(FRIBIDI_ADJUST_BITS((x), (mask), (cond)), (mask))

#include "fribidi-enddecls.h"

#endif /* !_FRIBIDI_TYPES_H */
/* Editor directions:
 * vim:textwidth=78:tabstop=8:shiftwidth=2:autoindent:cindent
 */
