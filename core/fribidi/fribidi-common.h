/* FriBidi
 * fribidi-common.h - common include for library headers
 *
 * $Id: fribidi-common.h,v 1.14 2010-02-24 19:40:04 behdad Exp $
 * $Author: behdad $
 * $Date: 2010-02-24 19:40:04 $
 * $Revision: 1.14 $
 * $Source: /home/behdad/src/fdo/fribidi/togit/git/../fribidi/fribidi2/lib/fribidi-common.h,v $
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
#ifndef _FRIBIDI_COMMON_H
#define _FRIBIDI_COMMON_H

#if DONT_HAVE_FRIBIDI_CONFIG_H+0
# define FRIBIDI "fribidi"
# define FRIBIDI_NAME "fribidi"
# define FRIBIDI_VERSION "unknown"
# define FRIBIDI_BUGREPORT "unknown"
# define FRIBIDI_INTERFACE_VERSION_STRING "unknown"
#else /* !DONT_HAVE_FRIBIDI_CONFIG_H */
# include "fribidi-config.h"
#endif /* !DONT_HAVE_FRIBIDI_CONFIG_H */

#if HAVE_FRIBIDI_CUSTOM_H+0
# include <fribidi-custom.h>
#endif /* HAVE_FRIBIDI_CUSTOM_H */

/* FRIBIDI_NAMESPACE is a macro used to name library symbols. */
#ifndef FRIBIDI_NAMESPACE
# define FRIBIDI_NAMESPACE(SYMBOL) fribidi##_##SYMBOL
#endif /* !FRIBIDI_NAMESPACE */

/* FRIBIDI_ENTRY is a macro used to declare library entry points. */
#ifndef FRIBIDI_ENTRY
#  define FRIBIDI_ENTRY		/* empty */
#endif /* !FRIBIDI_ENTRY */

#if FRIBIDI_USE_GLIB+0
# ifndef __FRIBIDI_DOC
#  include <glib.h>
# endif	/* !__FRIBIDI_DOC */
# define FRIBIDI_BEGIN_DECLS		G_BEGIN_DECLS
# define FRIBIDI_END_DECLS		G_END_DECLS
# define FRIBIDI_GNUC_CONST		G_GNUC_CONST
# define FRIBIDI_GNUC_DEPRECATED	G_GNUC_DEPRECATED
# if __GNUC__ > 2
#  define FRIBIDI_GNUC_WARN_UNUSED	\
	__attribute__((__warn_unused_result__))
#  define FRIBIDI_GNUC_MALLOC		\
	__attribute__((__malloc__))
#  define FRIBIDI_GNUC_HIDDEN		\
	__attribute__((__visibility__ ("hidden")))
# else /* __GNUC__ <= 2 */
#  define FRIBIDI_GNUC_WARN_UNUSED
#  define FRIBIDI_GNUC_MALLOC
#  define FRIBIDI_GNUC_HIDDEN
# endif	/* __GNUC__ <= 2 */
#else /* !FRIBIDI_USE_GLIB */
# define FRIBIDI_GNUC_CONST
# define FRIBIDI_GNUC_DEPRECATED
# define FRIBIDI_GNUC_WARN_UNUSED
# define FRIBIDI_GNUC_MALLOC
# define FRIBIDI_GNUC_HIDDEN
#endif /* !FRIBIDI_USE_GLIB */

/* FRIBIDI_BEGIN_DECLS should be used at the beginning of your declarations,
 * so that C++ compilers don't mangle their names.  Use FRIBIDI_END_DECLS at
 * the end of C declarations. */
#ifndef FRIBIDI_BEGIN_DECLS
# ifdef __cplusplus
#  define FRIBIDI_BEGIN_DECLS extern "C" {
#  define FRIBIDI_END_DECLS }
# else /* !__cplusplus */
#  define FRIBIDI_BEGIN_DECLS	/* empty */
#  define FRIBIDI_END_DECLS	/* empty */
# endif	/* !__cplusplus */
#endif /* !FRIBIDI_BEGIN_DECLS */




#define fribidi_debug_status FRIBIDI_NAMESPACE(debug_status)
FRIBIDI_ENTRY int fribidi_debug_status (
  void
);
#define fribidi_set_debug FRIBIDI_NAMESPACE(set_debug)
FRIBIDI_ENTRY int
fribidi_set_debug (
  int state		/* new state to set */
);











#endif /* !_FRIBIDI_COMMON_H */
/* Editor directions:
 * vim:textwidth=78:tabstop=8:shiftwidth=2:autoindent:cindent
 */
