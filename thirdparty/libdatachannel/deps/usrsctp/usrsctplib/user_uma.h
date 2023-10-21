/*-
 * Copyright (c) 2009-2010 Brad Penoff
 * Copyright (c) 2009-2010 Humaira Kamal
 * Copyright (c) 2011-2012 Irene Ruengeler
 * Copyright (c) 2011-2012 Michael Tuexen
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef _USER_UMA_H_
#define _USER_UMA_H_

#define UMA_ZFLAG_FULL		0x40000000	/* Reached uz_maxpages */
#define UMA_ALIGN_PTR	(sizeof(void *) - 1)	/* Alignment fit for ptr */

/* __Userspace__ All these definitions will change for
userspace Universal Memory Allocator (UMA). These are included
for reference purposes and to avoid compile errors for the time being.
*/
typedef int (*uma_ctor)(void *mem, int size, void *arg, int flags);
typedef void (*uma_dtor)(void *mem, int size, void *arg);
typedef int (*uma_init)(void *mem, int size, int flags);
typedef void (*uma_fini)(void *mem, int size);
typedef struct uma_zone * uma_zone_t;
typedef struct uma_keg * uma_keg_t;

struct uma_cache {
	int stub; /* TODO __Userspace__ */
};

struct uma_keg {
	int stub; /* TODO __Userspace__ */
};

struct uma_zone {
	char		*uz_name;	/* Text name of the zone */
	struct mtx	*uz_lock;	/* Lock for the zone (keg's lock) */
	uma_keg_t	uz_keg;		/* Our underlying Keg */

	LIST_ENTRY(uma_zone)	uz_link;	/* List of all zones in keg */
	LIST_HEAD(,uma_bucket)	uz_full_bucket;	/* full buckets */
	LIST_HEAD(,uma_bucket)	uz_free_bucket;	/* Buckets for frees */

	uma_ctor	uz_ctor;	/* Constructor for each allocation */
	uma_dtor	uz_dtor;	/* Destructor */
	uma_init	uz_init;	/* Initializer for each item */
	uma_fini	uz_fini;	/* Discards memory */

	u_int64_t	uz_allocs;	/* Total number of allocations */
	u_int64_t	uz_frees;	/* Total number of frees */
	u_int64_t	uz_fails;	/* Total number of alloc failures */
	uint16_t	uz_fills;	/* Outstanding bucket fills */
	uint16_t	uz_count;	/* Highest value ub_ptr can have */

	/*
	 * This HAS to be the last item because we adjust the zone size
	 * based on NCPU and then allocate the space for the zones.
	 */
	struct uma_cache	uz_cpu[1];	/* Per cpu caches */
};

/* Prototype */
uma_zone_t
uma_zcreate(char *name, size_t size, uma_ctor ctor, uma_dtor dtor,
	    uma_init uminit, uma_fini fini, int align, uint32_t flags);


#define uma_zone_set_max(zone, number) /* stub TODO __Userspace__ */

uma_zone_t
uma_zcreate(char *name, size_t size, uma_ctor ctor, uma_dtor dtor,
	    uma_init uminit, uma_fini fini, int align, uint32_t flags) {
	return NULL; /* stub TODO __Userspace__. Also place implementation in a separate .c file */
}
#endif
