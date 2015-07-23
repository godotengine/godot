/* FriBidi
 * fribidi-mem.c - memory manipulation routines
 *
 * $Id: fribidi-mem.c,v 1.8 2006-01-31 03:23:13 behdad Exp $
 * $Author: behdad $
 * $Date: 2006-01-31 03:23:13 $
 * $Revision: 1.8 $
 * $Source: /home/behdad/src/fdo/fribidi/togit/git/../fribidi/fribidi2/lib/fribidi-mem.c,v $
 *
 * Authors:
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

#include "common.h"

#include "mem.h"

#if FRIBIDI_USE_GLIB+0
#else
#if USE_SIMPLE_MALLOC+0
#else

struct _FriBidiMemChunk
{
  int atom_size;
  int area_size;
  int empty_size;
  void *chunk;
};

FriBidiMemChunk *
fribidi_mem_chunk_new (
  /* input */
  const char *name,
  int atom_size,
  unsigned long area_size,
  int alloc_type
)
{
  register FriBidiMemChunk *m;

  fribidi_assert (area_size >= atom_size * 8);

  m = (FriBidiMemChunk *) fribidi_malloc (sizeof (FriBidiMemChunk));
  if LIKELY
    (m)
    {
      m->atom_size = atom_size;
      m->area_size = area_size;
      m->empty_size = 0;
      m->chunk = NULL;
    }

  return m;
}

void *
fribidi_mem_chunk_alloc (
  /* input */
  FriBidiMemChunk *mem_chunk
)
{
  fribidi_assert (mem_chunk);

  if UNLIKELY
    (mem_chunk->empty_size < mem_chunk->atom_size)
    {
      register void *chunk = fribidi_malloc (mem_chunk->area_size);
      if LIKELY
	(chunk)
	{
	  if (mem_chunk->chunk)
	    *(void **) chunk =
	      (char *) mem_chunk->chunk + mem_chunk->empty_size -
	      mem_chunk->area_size;
	  chunk = (char *) chunk + mem_chunk->atom_size;
	  mem_chunk->chunk = chunk;
	  mem_chunk->empty_size = mem_chunk->area_size - mem_chunk->atom_size;
	}
      else
	return NULL;
    }

  {
    register void *m = mem_chunk->chunk;
    mem_chunk->chunk = (char *) mem_chunk->chunk + mem_chunk->atom_size;
    mem_chunk->empty_size -= mem_chunk->atom_size;

    return m;
  }
}

void
fribidi_mem_chunk_destroy (
  /* input */
  FriBidiMemChunk *mem_chunk
)
{
  register void *chunk;

  fribidi_assert (mem_chunk);

  chunk =
    (char *) mem_chunk->chunk + mem_chunk->empty_size - mem_chunk->area_size;
  while LIKELY
    (chunk)
    {
      register void *tofree = chunk;
      chunk = *(void **) chunk;
      fribidi_free (tofree);
    }
  fribidi_free (mem_chunk);
}

#endif /* !USE_SIMPLE_MALLOC */
#endif /* !FRIBIDI_USE_GLIB */

/* Editor directions:
 * vim:textwidth=78:tabstop=8:shiftwidth=2:autoindent:cindent
 */
