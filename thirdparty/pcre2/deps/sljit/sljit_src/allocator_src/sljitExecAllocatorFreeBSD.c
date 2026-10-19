/*
 *    Stack-less Just-In-Time compiler
 *
 *    Copyright Zoltan Herczeg (hzmester@freemail.hu). All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are
 * permitted provided that the following conditions are met:
 *
 *   1. Redistributions of source code must retain the above copyright notice, this list of
 *      conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright notice, this list
 *      of conditions and the following disclaimer in the documentation and/or other materials
 *      provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER(S) AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDER(S) OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <sys/mman.h>
#include <sys/procctl.h>

#ifdef PROC_WXMAP_CTL
static SLJIT_INLINE int sljit_is_wx_block(void)
{
	static int wx_block = -1;
	if (wx_block < 0) {
		int sljit_wx_enable = PROC_WX_MAPPINGS_PERMIT;
		wx_block = !!procctl(P_PID, 0, PROC_WXMAP_CTL, &sljit_wx_enable);
	}
	return wx_block;
}

#define SLJIT_IS_WX_BLOCK sljit_is_wx_block()
#else /* !PROC_WXMAP_CTL */
#define SLJIT_IS_WX_BLOCK (1)
#endif /* PROC_WXMAP_CTL */

static SLJIT_INLINE void* alloc_chunk(sljit_uw size)
{
	void *retval;
	int prot = PROT_READ | PROT_WRITE | PROT_EXEC;
	int flags = MAP_PRIVATE;
	int fd = -1;

#ifdef PROT_MAX
	prot |= PROT_MAX(prot);
#endif

#ifdef MAP_ANON
	flags |= MAP_ANON;
#else /* !MAP_ANON */
	if (SLJIT_UNLIKELY((dev_zero < 0) && open_dev_zero()))
		return NULL;

	fd = dev_zero;
#endif /* MAP_ANON */

retry:
	retval = mmap(NULL, size, prot, flags, fd, 0);
	if (retval == MAP_FAILED) {
		if (!SLJIT_IS_WX_BLOCK)
			goto retry;

		return NULL;
	}

	/* HardenedBSD's mmap lies, so check permissions again. */
	if (mprotect(retval, size, PROT_READ | PROT_WRITE | PROT_EXEC) < 0) {
		munmap(retval, size);
		return NULL;
	}

	return retval;
}

static SLJIT_INLINE void free_chunk(void *chunk, sljit_uw size)
{
	munmap(chunk, size);
}

#include "sljitExecAllocatorCore.c"
