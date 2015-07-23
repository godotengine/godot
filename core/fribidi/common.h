/* FriBidi
 * common.h - common include for library sources
 *
 * $Id: common.h,v 1.21 2010-02-24 19:40:04 behdad Exp $
 * $Author: behdad $
 * $Date: 2010-02-24 19:40:04 $
 * $Revision: 1.21 $
 * $Source: /home/behdad/src/fdo/fribidi/togit/git/../fribidi/fribidi2/lib/common.h,v $
 *
 * Author:
 *   Behdad Esfahbod, 2004
 *
 * Copyright (C) 2004 Sharif FarsiWeb, Inc.
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
#ifndef _COMMON_H
#define _COMMON_H

#if HAVE_CONFIG_H+0
# include "config.h"
#endif

#include "fribidi-common.h"

/* FRIBIDI_PRIVATESPACE is a macro used to name library internal symbols. */
#ifndef FRIBIDI_PRIVATESPACE
# define FRIBIDI_PRIVATESPACE1(A,B) A##B
# define FRIBIDI_PRIVATESPACE0(A,B) FRIBIDI_PRIVATESPACE1(A,B)
# define FRIBIDI_PRIVATESPACE(SYMBOL) FRIBIDI_PRIVATESPACE0(_,FRIBIDI_NAMESPACE(_##SYMBOL##__internal__))
#endif /* !FRIBIDI_PRIVATESPACE */

#if FRIBIDI_USE_GLIB+0
# ifndef SIZEOF_LONG
#  define SIZEOF_LONG GLIB_SIZEOF_LONG
# endif	/* !SIZEOF_LONG */
# ifndef SIZEOF_VOID_P
#  define SIZEOF_VOID_P GLIB_SIZEOF_VOID_P
# endif	/* !SIZEOF_VOID_P */
# ifndef __FRIBIDI_DOC
#  include <glib.h>
# endif	/* !__FRIBIDI_DOC */
# ifndef fribidi_malloc
#  define fribidi_malloc g_try_malloc
#  define fribidi_free g_free
# endif	/* !fribidi_malloc */
# ifndef fribidi_assert
#  ifndef __FRIBIDI_DOC
#   include <glib.h>
#  endif /* !__FRIBIDI_DOC */
#  define fribidi_assert g_assert
# endif	/* !fribidi_assert */
# ifndef __FRIBIDI_DOC
#  include <glib.h>
# endif	/* !__FRIBIDI_DOC */
# ifndef FRIBIDI_BEGIN_STMT
#  define FRIBIDI_BEGIN_STMT G_STMT_START {
#  define FRIBIDI_END_STMT } G_STMT_END
# endif	/* !FRIBIDI_BEGIN_STMT */
# ifndef LIKELY
#  define LIKELY G_LIKELY
#  define UNLIKELY G_UNLIKELY
# endif	/* !LIKELY */
# ifndef false
#  define false FALSE
# endif	/* !false */
# ifndef true
#  define true TRUE
# endif	/* !true */
#endif /* FRIBIDI_USE_GLIB */

#ifndef false
# define false (0)
# endif	/* !false */
# ifndef true
#  define true (!false)
# endif	/* !true */

#ifndef NULL
#  ifdef __cplusplus
#    define NULL        (0L)
#  else	/* !__cplusplus */
#    define NULL        ((void*) 0)
#  endif /* !__cplusplus */
#endif /* !NULL */

/* fribidi_malloc and fribidi_free should be used instead of malloc and free. 
 * No need to include any headers. */
#ifndef fribidi_malloc
# if HAVE_STDLIB_H
#  ifndef __FRIBIDI_DOC
#   include <stdlib.h>
#  endif /* __FRIBIDI_DOC */
#  define fribidi_malloc malloc
# else /* !HAVE_STDLIB_H */
#  define fribidi_malloc (void *) malloc
# endif	/* !HAVE_STDLIB_H */
# define fribidi_free free
#else /* fribidi_malloc */
# ifndef fribidi_free
#  error You should define fribidi_free too when you define fribidi_malloc.
# endif	/* !fribidi_free */
#endif /* fribidi_malloc */

#if HAVE_STRING_H+0
# if !STDC_HEADERS && HAVE_MEMORY_H
#  include <memory.h>
# endif
# include <string.h>
#endif
#if HAVE_STRINGS_H+0
# include <strings.h>
#endif

/* FRIBIDI_CHUNK_SIZE is the number of bytes in each chunk of memory being
 * allocated for data structure pools. */
#ifndef FRIBIDI_CHUNK_SIZE
# if HAVE_ASM_PAGE_H
#  ifndef __FRIBIDI_DOC
#   include <asm/page.h>
#  endif /* __FRIBIDI_DOC */
#  define FRIBIDI_CHUNK_SIZE (PAGE_SIZE - 16)
# else /* !HAVE_ASM_PAGE_H */
#  define FRIBIDI_CHUNK_SIZE (4096 - 16)
# endif	/* !HAVE_ASM_PAGE_H */
#else /* FRIBIDI_CHUNK_SIZE */
# if FRIBIDI_CHUNK_SIZE < 256
#  error FRIBIDI_CHUNK_SIZE now should define the size of a chunk in bytes.
# endif	/* FRIBIDI_CHUNK_SIZE < 256 */
#endif /* FRIBIDI_CHUNK_SIZE */

/* FRIBIDI_BEGIN_STMT should be used at the beginning of your macro
 * definitions that are to behave like simple statements.  Use
 * FRIBIDI_END_STMT at the end of the macro after the semicolon or brace. */
#ifndef FRIBIDI_BEGIN_STMT
# define FRIBIDI_BEGIN_STMT do {
# define FRIBIDI_END_STMT } while (0)
#endif /* !FRIBIDI_BEGIN_STMT */

/* LIKEYLY and UNLIKELY are used to give a hint on branch prediction to the
 * compiler. */
#ifndef LIKELY
# define LIKELY
# define UNLIKELY
#endif /* !LIKELY */

#ifndef FRIBIDI_EMPTY_STMT
# define FRIBIDI_EMPTY_STMT FRIBIDI_BEGIN_STMT (void) 0; FRIBIDI_END_STMT
#endif /* !FRIBIDI_EMPTY_STMT */

#if HAVE_STRINGIZE+0
# define STRINGIZE(symbol) #symbol
#else /* !HAVE_STRINGIZE */
# define STRINGIZE(symbol) (no stringize operator available)
#endif /* !HAVE_STRINGIZE */

/* As per recommendation of GNU Coding Standards. */
#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif /* !_GNU_SOURCE */

/* We respect our own rules. */
#define FRIBIDI_NO_DEPRECATED


#include "debug.h"

#endif /* !_COMMON_H */
/* Editor directions:
 * vim:textwidth=78:tabstop=8:shiftwidth=2:autoindent:cindent
 */
