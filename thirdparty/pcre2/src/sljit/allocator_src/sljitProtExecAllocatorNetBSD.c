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

#define SLJIT_HAS_CHUNK_HEADER
#define SLJIT_HAS_EXECUTABLE_OFFSET

struct sljit_chunk_header {
	void *executable;
};

/*
 * MAP_REMAPDUP is a NetBSD extension available sinde 8.0, make sure to
 * adjust your feature macros (ex: -D_NETBSD_SOURCE) as needed
 */
static SLJIT_INLINE struct sljit_chunk_header* alloc_chunk(sljit_uw size)
{
	struct sljit_chunk_header *retval;

	retval = (struct sljit_chunk_header *)mmap(NULL, size,
			PROT_READ | PROT_WRITE | PROT_MPROTECT(PROT_EXEC),
			MAP_ANON | MAP_SHARED, -1, 0);

	if (retval == MAP_FAILED)
		return NULL;

	retval->executable = mremap(retval, size, NULL, size, MAP_REMAPDUP);
	if (retval->executable == MAP_FAILED) {
		munmap((void *)retval, size);
		return NULL;
	}

	if (mprotect(retval->executable, size, PROT_READ | PROT_EXEC) == -1) {
		munmap(retval->executable, size);
		munmap((void *)retval, size);
		return NULL;
	}

	return retval;
}

static SLJIT_INLINE void free_chunk(void *chunk, sljit_uw size)
{
	struct sljit_chunk_header *header = ((struct sljit_chunk_header *)chunk) - 1;

	munmap(header->executable, size);
	munmap((void *)header, size);
}

#include "sljitExecAllocatorCore.c"
