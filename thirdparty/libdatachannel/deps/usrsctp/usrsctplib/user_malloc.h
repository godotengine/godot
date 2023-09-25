/*-
 * Copyright (c) 1987, 1993
 *	The Regents of the University of California.
 * Copyright (c) 2005 Robert N. M. Watson
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
 * 3. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 */

/* This file has been renamed user_malloc.h for Userspace */
#ifndef _USER_MALLOC_H_
#define	_USER_MALLOC_H_

/*__Userspace__*/
#include <stdlib.h>
#include <sys/types.h>
#if !defined(_WIN32)
#include <strings.h>
#include <stdint.h>
#else
#if (defined(_MSC_VER) && _MSC_VER >= 1600) || (defined(__MSVCRT_VERSION__) && __MSVCRT_VERSION__ >= 1400)
#include <stdint.h>
#elif defined(SCTP_STDINT_INCLUDE)
#include SCTP_STDINT_INCLUDE
#else
#define uint32_t unsigned __int32
#define uint64_t unsigned __int64
#endif
#include <winsock2.h>
#endif

#define	MINALLOCSIZE	UMA_SMALLEST_UNIT

/*
 * flags to malloc.
 */
#define	M_NOWAIT	0x0001		/* do not block */
#define	M_WAITOK	0x0002		/* ok to block */
#define	M_ZERO		0x0100		/* bzero the allocation */
#define	M_NOVM		0x0200		/* don't ask VM for pages */
#define	M_USE_RESERVE	0x0400		/* can alloc out of reserve memory */

#define	M_MAGIC		877983977	/* time when first defined :-) */

/*
 * Two malloc type structures are present: malloc_type, which is used by a
 * type owner to declare the type, and malloc_type_internal, which holds
 * malloc-owned statistics and other ABI-sensitive fields, such as the set of
 * malloc statistics indexed by the compile-time MAXCPU constant.
 * Applications should avoid introducing dependence on the allocator private
 * data layout and size.
 *
 * The malloc_type ks_next field is protected by malloc_mtx.  Other fields in
 * malloc_type are static after initialization so unsynchronized.
 *
 * Statistics in malloc_type_stats are written only when holding a critical
 * section and running on the CPU associated with the index into the stat
 * array, but read lock-free resulting in possible (minor) races, which the
 * monitoring app should take into account.
 */
struct malloc_type_stats {
	uint64_t	mts_memalloced;	/* Bytes allocated on CPU. */
	uint64_t	mts_memfreed;	/* Bytes freed on CPU. */
	uint64_t	mts_numallocs;	/* Number of allocates on CPU. */
	uint64_t	mts_numfrees;	/* number of frees on CPU. */
	uint64_t	mts_size;	/* Bitmask of sizes allocated on CPU. */
	uint64_t	_mts_reserved1;	/* Reserved field. */
	uint64_t	_mts_reserved2;	/* Reserved field. */
	uint64_t	_mts_reserved3;	/* Reserved field. */
};

#ifndef MAXCPU /* necessary on Linux */
#define MAXCPU 4 /* arbitrary? */
#endif

struct malloc_type_internal {
	struct malloc_type_stats	mti_stats[MAXCPU];
};

/*
 * ABI-compatible version of the old 'struct malloc_type', only all stats are
 * now malloc-managed in malloc-owned memory rather than in caller memory, so
 * as to avoid ABI issues.  The ks_next pointer is reused as a pointer to the
 * internal data handle.
 */
struct malloc_type {
	struct malloc_type *ks_next;	/* Next in global chain. */
	u_long		 _ks_memuse;	/* No longer used. */
	u_long		 _ks_size;	/* No longer used. */
	u_long		 _ks_inuse;	/* No longer used. */
	uint64_t	 _ks_calls;	/* No longer used. */
	u_long		 _ks_maxused;	/* No longer used. */
	u_long		 ks_magic;	/* Detect programmer error. */
	const char	*ks_shortdesc;	/* Printable type name. */

	/*
	 * struct malloc_type was terminated with a struct mtx, which is no
	 * longer required.  For ABI reasons, continue to flesh out the full
	 * size of the old structure, but reuse the _lo_class field for our
	 * internal data handle.
	 */
	void		*ks_handle;	/* Priv. data, was lo_class. */
	const char	*_lo_name;
	const char	*_lo_type;
	u_int		 _lo_flags;
	void		*_lo_list_next;
	struct witness	*_lo_witness;
	uintptr_t	 _mtx_lock;
	u_int		 _mtx_recurse;
};

/*
 * Statistics structure headers for user space.  The kern.malloc sysctl
 * exposes a structure stream consisting of a stream header, then a series of
 * malloc type headers and statistics structures (quantity maxcpus).  For
 * convenience, the kernel will provide the current value of maxcpus at the
 * head of the stream.
 */
#define	MALLOC_TYPE_STREAM_VERSION	0x00000001
struct malloc_type_stream_header {
	uint32_t	mtsh_version;	/* Stream format version. */
	uint32_t	mtsh_maxcpus;	/* Value of MAXCPU for stream. */
	uint32_t	mtsh_count;	/* Number of records. */
	uint32_t	_mtsh_pad;	/* Pad/reserved field. */
};

#define	MALLOC_MAX_NAME	32
struct malloc_type_header {
	char				mth_name[MALLOC_MAX_NAME];
};

/* __Userspace__
Notice that at places it uses ifdef _KERNEL. That line cannot be
removed because it causes conflicts with malloc definition in
/usr/include/malloc.h, which essentially says that malloc.h has
been overridden by stdlib.h. We will need to use names like
user_malloc.h for isolating kernel interface headers. using
original names like malloc.h in a user_include header can be
confusing, All userspace header files are being placed in ./user_include
Better still to remove from user_include.h all irrelevant code such
as that in the block starting with #ifdef _KERNEL. I am only leaving
it in for the time being to see what functionality is in this file
that kernel uses.

Start copy: Copied code for __Userspace__ */
#define	MALLOC_DEFINE(type, shortdesc, longdesc)			\
	struct malloc_type type[1] = {					\
		{ NULL, 0, 0, 0, 0, 0, M_MAGIC, shortdesc, NULL, NULL,	\
		    NULL, 0, NULL, NULL, 0, 0 }				\
	}

/* Removed "extern" in __Userspace__ code */
/* If we need to use MALLOC_DECLARE before using MALLOC then
   we have to remove extern.
     In /usr/include/sys/malloc.h there is this definition:
     #define    MALLOC_DECLARE(type) \
        extern struct malloc_type type[1]
     and loader is unable to find the extern malloc_type because
     it may be defined in one of kernel object files.
     It seems that MALLOC_DECLARE and MALLOC_DEFINE cannot be used at
     the same time for same "type" variable. Also, in Randall's architecture
     document, where it specifies O/S specific macros and functions, it says
     that the name in SCTP_MALLOC does not have to be used.
*/
#define	MALLOC_DECLARE(type) \
	extern struct malloc_type type[1]

#define	FREE(addr, type) free((addr))

/* changed definitions of MALLOC and FREE */
/* Using memset if flag M_ZERO is specified. Todo: M_WAITOK and M_NOWAIT */
#define	MALLOC(space, cast, size, type, flags)                          \
    ((space) = (cast)malloc((u_long)(size)));                           \
    do {								\
        if (flags & M_ZERO) {                                            \
	  memset(space,0,size);                                         \
	}								\
    } while (0);

#endif /* !_SYS_MALLOC_H_ */
