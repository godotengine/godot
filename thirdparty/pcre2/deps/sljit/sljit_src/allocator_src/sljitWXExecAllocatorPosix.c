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

/*
   This file contains a simple W^X executable memory allocator

   In *NIX, MAP_ANON is required (that is considered a feature) so make
   sure to set the right availability macros for your system or the code
   will fail to build.

   If your system doesn't support mapping of anonymous pages (ex: IRIX) it
   is also likely that it doesn't need this allocator and should be using
   the standard one instead.

   It allocates a separate map for each code block and may waste a lot of
   memory, because whatever was requested, will be rounded up to the page
   size (minimum 4KB, but could be even bigger).

   It changes the page permissions (RW <-> RX) as needed and therefore, if you
   will be updating the code after it has been generated, need to make sure to
   block any concurrent execution, or could result in a SIGBUS, that could
   even manifest itself at a different address than the one that was being
   modified.

   Only use if you are unable to use the regular allocator because of security
   restrictions and adding exceptions to your application or the system are
   not possible.
*/

#include <sys/types.h>
#include <sys/mman.h>

#define SLJIT_UPDATE_WX_FLAGS(from, to, enable_exec) \
	sljit_update_wx_flags((from), (to), (enable_exec))

#if !(defined SLJIT_SINGLE_THREADED && SLJIT_SINGLE_THREADED)
#include <pthread.h>
#define SLJIT_SE_LOCK()		pthread_mutex_lock(&se_lock)
#define SLJIT_SE_UNLOCK()	pthread_mutex_unlock(&se_lock)
#else
#define SLJIT_SE_LOCK()
#define SLJIT_SE_UNLOCK()
#endif /* !SLJIT_SINGLE_THREADED */

#define SLJIT_WX_IS_BLOCK(ptr, size) generic_check_is_wx_block(ptr, size)

static SLJIT_INLINE int generic_check_is_wx_block(void *ptr, sljit_uw size)
{
	if (SLJIT_LIKELY(!mprotect(ptr, size, PROT_EXEC)))
		return !!mprotect(ptr, size, PROT_READ | PROT_WRITE);

	return 1;
}

SLJIT_API_FUNC_ATTRIBUTE void* sljit_malloc_exec(sljit_uw size)
{
#if !(defined SLJIT_SINGLE_THREADED && SLJIT_SINGLE_THREADED)
	static pthread_mutex_t se_lock = PTHREAD_MUTEX_INITIALIZER;
#endif
	static int wx_block = -1;
	int prot = PROT_READ | PROT_WRITE;
	sljit_uw* ptr;

	if (SLJIT_UNLIKELY(wx_block > 0))
		return NULL;

#ifdef PROT_MAX
	prot |= PROT_MAX(PROT_READ | PROT_WRITE | PROT_EXEC);
#endif

	size += sizeof(sljit_uw);
	ptr = (sljit_uw*)mmap(NULL, size, prot, MAP_PRIVATE | MAP_ANON, -1, 0);

	if (ptr == MAP_FAILED)
		return NULL;

	if (SLJIT_UNLIKELY(wx_block < 0)) {
		SLJIT_SE_LOCK();
		wx_block = SLJIT_WX_IS_BLOCK(ptr, size);
		SLJIT_SE_UNLOCK();
		if (SLJIT_UNLIKELY(wx_block)) {
			munmap((void *)ptr, size);
			return NULL;
		}
	}

	*ptr++ = size;
	return ptr;
}

#undef SLJIT_SE_UNLOCK
#undef SLJIT_SE_LOCK

SLJIT_API_FUNC_ATTRIBUTE void sljit_free_exec(void* ptr)
{
	sljit_uw *start_ptr = ((sljit_uw*)ptr) - 1;
	munmap((void*)start_ptr, *start_ptr);
}

static void sljit_update_wx_flags(void *from, void *to, int enable_exec)
{
	sljit_uw page_mask = (sljit_uw)get_page_alignment();
	sljit_uw start = (sljit_uw)from;
	sljit_uw end = (sljit_uw)to;
	int prot = PROT_READ | (enable_exec ? PROT_EXEC : PROT_WRITE);

	SLJIT_ASSERT(start < end);

	start &= ~page_mask;
	end = (end + page_mask) & ~page_mask;

	mprotect((void*)start, end - start, prot);
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_free_unused_memory_exec(void)
{
	/* This allocator does not keep unused memory for future allocations. */
}
