/* FriBidi
 * mem.h - memory manipulation routines
 *
 * $Id: mem.h,v 1.7 2006-01-31 03:23:13 behdad Exp $
 * $Author: behdad $
 * $Date: 2006-01-31 03:23:13 $
 * $Revision: 1.7 $
 * $Source: /home/behdad/src/fdo/fribidi/togit/git/../fribidi/fribidi2/lib/mem.h,v $
 *
 * Author:
 *   Behdad Esfahbod, 2001, 2002, 2004
 *
 * Copyright (C) 2004 Sharif FarsiWeb, Inc
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
#ifndef _MEM_H
#define _MEM_H

#include "common.h"

#include <fribidi-types.h>

#include <fribidi-begindecls.h>

#if FRIBIDI_USE_GLIB+0

#ifndef __FRIBIDI_DOC
# include <glib/gmem.h>
#endif /* !__FRIBIDI_DOC */

#define FriBidiMemChunk GMemChunk
#define FRIBIDI_ALLOC_ONLY G_ALLOC_ONLY
#define fribidi_mem_chunk_new g_mem_chunk_new
#define fribidi_mem_chunk_alloc g_mem_chunk_alloc
#define fribidi_mem_chunk_destroy g_mem_chunk_destroy

#else /* !FRIBIDI_USE_GLIB */

typedef struct _FriBidiMemChunk FriBidiMemChunk;

#define FRIBIDI_ALLOC_ONLY      1

#define fribidi_mem_chunk_new FRIBIDI_PRIVATESPACE(mem_chunk_new)
FriBidiMemChunk *
fribidi_mem_chunk_new (
  const char *name,
  int atom_size,
  unsigned long area_size,
  int alloc_type
)
     FRIBIDI_GNUC_HIDDEN FRIBIDI_GNUC_MALLOC FRIBIDI_GNUC_WARN_UNUSED;

#define fribidi_mem_chunk_alloc FRIBIDI_PRIVATESPACE(mem_chunk_alloc)
     void *fribidi_mem_chunk_alloc (
  FriBidiMemChunk *mem_chunk
)
     FRIBIDI_GNUC_HIDDEN FRIBIDI_GNUC_MALLOC FRIBIDI_GNUC_WARN_UNUSED;

#define fribidi_mem_chunk_destroy FRIBIDI_PRIVATESPACE(mem_chunk_destroy)
     void fribidi_mem_chunk_destroy (
  FriBidiMemChunk *mem_chunk
) FRIBIDI_GNUC_HIDDEN;

#endif /* !FRIBIDI_USE_GLIB */

#define fribidi_chunk_new(type, chunk)        ( \
		(type *) fribidi_mem_chunk_alloc (chunk) \
	)

#define fribidi_chunk_new_for_type(type) ( \
		fribidi_mem_chunk_new(FRIBIDI, sizeof (type), \
				FRIBIDI_CHUNK_SIZE, FRIBIDI_ALLOC_ONLY) \
	)

#include <fribidi-enddecls.h>

#endif /* !_MEM_H */
/* Editor directions:
 * vim:textwidth=78:tabstop=8:shiftwidth=2:autoindent:cindent
 */
