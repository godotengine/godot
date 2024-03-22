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

#include <sys/types.h>
#include <sys/mman.h>
/*
   On macOS systems, returns MAP_JIT if it is defined _and_ we're running on a
   version where it's OK to have more than one JIT block or where MAP_JIT is
   required.
   On non-macOS systems, returns MAP_JIT if it is defined.
*/
#include <TargetConditionals.h>

#if (defined(TARGET_OS_OSX) && TARGET_OS_OSX) || (TARGET_OS_MAC && !TARGET_OS_IPHONE)

#if defined(SLJIT_CONFIG_X86) && SLJIT_CONFIG_X86

#include <sys/utsname.h>
#include <stdlib.h>

#define SLJIT_MAP_JIT	(get_map_jit_flag())
#define SLJIT_UPDATE_WX_FLAGS(from, to, enable_exec)

static SLJIT_INLINE int get_map_jit_flag(void)
{
	size_t page_size;
	void *ptr;
	struct utsname name;
	static int map_jit_flag = -1;

	if (map_jit_flag < 0) {
		map_jit_flag = 0;
		uname(&name);

		/* Kernel version for 10.14.0 (Mojave) or later */
		if (atoi(name.release) >= 18) {
			page_size = get_page_alignment() + 1;
			/* Only use MAP_JIT if a hardened runtime is used */
			ptr = mmap(NULL, page_size, PROT_WRITE | PROT_EXEC,
					MAP_PRIVATE | MAP_ANON, -1, 0);

			if (ptr != MAP_FAILED)
				munmap(ptr, page_size);
			else
				map_jit_flag = MAP_JIT;
		}
	}
	return map_jit_flag;
}

#elif defined(SLJIT_CONFIG_ARM) && SLJIT_CONFIG_ARM

#include <AvailabilityMacros.h>
#include <pthread.h>

#define SLJIT_MAP_JIT	(MAP_JIT)
#define SLJIT_UPDATE_WX_FLAGS(from, to, enable_exec) \
		apple_update_wx_flags(enable_exec)

static SLJIT_INLINE void apple_update_wx_flags(sljit_s32 enable_exec)
{
#if MAC_OS_X_VERSION_MIN_REQUIRED < 110000
	if (__builtin_available(macos 11, *))
#endif /* BigSur */
	pthread_jit_write_protect_np(enable_exec);
}

#elif defined(SLJIT_CONFIG_PPC) && SLJIT_CONFIG_PPC

#define SLJIT_MAP_JIT	(0)
#define SLJIT_UPDATE_WX_FLAGS(from, to, enable_exec)

#else
#error "Unsupported architecture"
#endif /* SLJIT_CONFIG */

#else /* !TARGET_OS_OSX */

#ifdef MAP_JIT
#define SLJIT_MAP_JIT	(MAP_JIT)
#else
#define SLJIT_MAP_JIT	(0)
#endif

#endif /* TARGET_OS_OSX */

static SLJIT_INLINE void* alloc_chunk(sljit_uw size)
{
	void *retval;
	int prot = PROT_READ | PROT_WRITE | PROT_EXEC;
	int flags = MAP_PRIVATE;
	int fd = -1;

	flags |= MAP_ANON | SLJIT_MAP_JIT;

	retval = mmap(NULL, size, prot, flags, fd, 0);
	if (retval == MAP_FAILED)
		return NULL;

	SLJIT_UPDATE_WX_FLAGS(retval, (uint8_t *)retval + size, 0);

	return retval;
}

static SLJIT_INLINE void free_chunk(void *chunk, sljit_uw size)
{
	munmap(chunk, size);
}

#include "sljitExecAllocatorCore.c"
