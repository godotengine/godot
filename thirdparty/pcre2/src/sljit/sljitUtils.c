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

/* ------------------------------------------------------------------------ */
/*  Locks                                                                   */
/* ------------------------------------------------------------------------ */

#if (defined SLJIT_EXECUTABLE_ALLOCATOR && SLJIT_EXECUTABLE_ALLOCATOR) || (defined SLJIT_UTIL_GLOBAL_LOCK && SLJIT_UTIL_GLOBAL_LOCK)

#if (defined SLJIT_SINGLE_THREADED && SLJIT_SINGLE_THREADED)

#if (defined SLJIT_EXECUTABLE_ALLOCATOR && SLJIT_EXECUTABLE_ALLOCATOR)

static SLJIT_INLINE void allocator_grab_lock(void)
{
	/* Always successful. */
}

static SLJIT_INLINE void allocator_release_lock(void)
{
	/* Always successful. */
}

#endif /* SLJIT_EXECUTABLE_ALLOCATOR */

#if (defined SLJIT_UTIL_GLOBAL_LOCK && SLJIT_UTIL_GLOBAL_LOCK)

SLJIT_API_FUNC_ATTRIBUTE void SLJIT_FUNC sljit_grab_lock(void)
{
	/* Always successful. */
}

SLJIT_API_FUNC_ATTRIBUTE void SLJIT_FUNC sljit_release_lock(void)
{
	/* Always successful. */
}

#endif /* SLJIT_UTIL_GLOBAL_LOCK */

#elif defined(_WIN32) /* SLJIT_SINGLE_THREADED */

#include "windows.h"

#if (defined SLJIT_EXECUTABLE_ALLOCATOR && SLJIT_EXECUTABLE_ALLOCATOR)

static HANDLE allocator_mutex = 0;

static SLJIT_INLINE void allocator_grab_lock(void)
{
	/* No idea what to do if an error occures. Static mutexes should never fail... */
	if (!allocator_mutex)
		allocator_mutex = CreateMutex(NULL, TRUE, NULL);
	else
		WaitForSingleObject(allocator_mutex, INFINITE);
}

static SLJIT_INLINE void allocator_release_lock(void)
{
	ReleaseMutex(allocator_mutex);
}

#endif /* SLJIT_EXECUTABLE_ALLOCATOR */

#if (defined SLJIT_UTIL_GLOBAL_LOCK && SLJIT_UTIL_GLOBAL_LOCK)

static HANDLE global_mutex = 0;

SLJIT_API_FUNC_ATTRIBUTE void SLJIT_FUNC sljit_grab_lock(void)
{
	/* No idea what to do if an error occures. Static mutexes should never fail... */
	if (!global_mutex)
		global_mutex = CreateMutex(NULL, TRUE, NULL);
	else
		WaitForSingleObject(global_mutex, INFINITE);
}

SLJIT_API_FUNC_ATTRIBUTE void SLJIT_FUNC sljit_release_lock(void)
{
	ReleaseMutex(global_mutex);
}

#endif /* SLJIT_UTIL_GLOBAL_LOCK */

#else /* _WIN32 */

#if (defined SLJIT_EXECUTABLE_ALLOCATOR && SLJIT_EXECUTABLE_ALLOCATOR)

#include <pthread.h>

static pthread_mutex_t allocator_mutex = PTHREAD_MUTEX_INITIALIZER;

static SLJIT_INLINE void allocator_grab_lock(void)
{
	pthread_mutex_lock(&allocator_mutex);
}

static SLJIT_INLINE void allocator_release_lock(void)
{
	pthread_mutex_unlock(&allocator_mutex);
}

#endif /* SLJIT_EXECUTABLE_ALLOCATOR */

#if (defined SLJIT_UTIL_GLOBAL_LOCK && SLJIT_UTIL_GLOBAL_LOCK)

#include <pthread.h>

static pthread_mutex_t global_mutex = PTHREAD_MUTEX_INITIALIZER;

SLJIT_API_FUNC_ATTRIBUTE void SLJIT_FUNC sljit_grab_lock(void)
{
	pthread_mutex_lock(&global_mutex);
}

SLJIT_API_FUNC_ATTRIBUTE void SLJIT_FUNC sljit_release_lock(void)
{
	pthread_mutex_unlock(&global_mutex);
}

#endif /* SLJIT_UTIL_GLOBAL_LOCK */

#endif /* _WIN32 */

/* ------------------------------------------------------------------------ */
/*  Stack                                                                   */
/* ------------------------------------------------------------------------ */

#if (defined SLJIT_UTIL_STACK && SLJIT_UTIL_STACK) || (defined SLJIT_EXECUTABLE_ALLOCATOR && SLJIT_EXECUTABLE_ALLOCATOR)

#ifdef _WIN32
#include "windows.h"
#else
/* Provides mmap function. */
#include <sys/types.h>
#include <sys/mman.h>
#ifndef MAP_ANON
#ifdef MAP_ANONYMOUS
#define MAP_ANON MAP_ANONYMOUS
#endif
#endif
/* For detecting the page size. */
#include <unistd.h>

#ifndef MAP_ANON

#include <fcntl.h>

/* Some old systems does not have MAP_ANON. */
static sljit_s32 dev_zero = -1;

#if (defined SLJIT_SINGLE_THREADED && SLJIT_SINGLE_THREADED)

static SLJIT_INLINE sljit_s32 open_dev_zero(void)
{
	dev_zero = open("/dev/zero", O_RDWR);
	return dev_zero < 0;
}

#else /* SLJIT_SINGLE_THREADED */

#include <pthread.h>

static pthread_mutex_t dev_zero_mutex = PTHREAD_MUTEX_INITIALIZER;

static SLJIT_INLINE sljit_s32 open_dev_zero(void)
{
	pthread_mutex_lock(&dev_zero_mutex);
	/* The dev_zero might be initialized by another thread during the waiting. */
	if (dev_zero < 0) {
		dev_zero = open("/dev/zero", O_RDWR);
	}
	pthread_mutex_unlock(&dev_zero_mutex);
	return dev_zero < 0;
}

#endif /* SLJIT_SINGLE_THREADED */

#endif

#endif

#endif /* SLJIT_UTIL_STACK || SLJIT_EXECUTABLE_ALLOCATOR */

#if (defined SLJIT_UTIL_STACK && SLJIT_UTIL_STACK)

/* Planning to make it even more clever in the future. */
static sljit_sw sljit_page_align = 0;

SLJIT_API_FUNC_ATTRIBUTE struct sljit_stack* SLJIT_FUNC sljit_allocate_stack(sljit_uw start_size, sljit_uw max_size, void *allocator_data)
{
	struct sljit_stack *stack;
	void *ptr;
#ifdef _WIN32
	SYSTEM_INFO si;
#endif

	SLJIT_UNUSED_ARG(allocator_data);
	if (start_size > max_size || start_size < 1)
		return NULL;

#ifdef _WIN32
	if (!sljit_page_align) {
		GetSystemInfo(&si);
		sljit_page_align = si.dwPageSize - 1;
	}
#else
	if (!sljit_page_align) {
		sljit_page_align = sysconf(_SC_PAGESIZE);
		/* Should never happen. */
		if (sljit_page_align < 0)
			sljit_page_align = 4096;
		sljit_page_align--;
	}
#endif

	stack = (struct sljit_stack*)SLJIT_MALLOC(sizeof(struct sljit_stack), allocator_data);
	if (!stack)
		return NULL;

	/* Align max_size. */
	max_size = (max_size + sljit_page_align) & ~sljit_page_align;

#ifdef _WIN32
	ptr = VirtualAlloc(NULL, max_size, MEM_RESERVE, PAGE_READWRITE);
	if (!ptr) {
		SLJIT_FREE(stack, allocator_data);
		return NULL;
	}

	stack->min_start = (sljit_u8 *)ptr;
	stack->end = stack->min_start + max_size;
	stack->start = stack->end;

	if (sljit_stack_resize(stack, stack->end - start_size) == NULL) {
		sljit_free_stack(stack, allocator_data);
		return NULL;
	}
#else
#ifdef MAP_ANON
	ptr = mmap(NULL, max_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
#else
	if (dev_zero < 0) {
		if (open_dev_zero()) {
			SLJIT_FREE(stack, allocator_data);
			return NULL;
		}
	}
	ptr = mmap(NULL, max_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, dev_zero, 0);
#endif
	if (ptr == MAP_FAILED) {
		SLJIT_FREE(stack, allocator_data);
		return NULL;
	}
	stack->min_start = (sljit_u8 *)ptr;
	stack->end = stack->min_start + max_size;
	stack->start = stack->end - start_size;
#endif
	stack->top = stack->end;
	return stack;
}

#undef PAGE_ALIGN

SLJIT_API_FUNC_ATTRIBUTE void SLJIT_FUNC sljit_free_stack(struct sljit_stack *stack, void *allocator_data)
{
	SLJIT_UNUSED_ARG(allocator_data);
#ifdef _WIN32
	VirtualFree((void*)stack->min_start, 0, MEM_RELEASE);
#else
	munmap((void*)stack->min_start, stack->end - stack->min_start);
#endif
	SLJIT_FREE(stack, allocator_data);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_u8 *SLJIT_FUNC sljit_stack_resize(struct sljit_stack *stack, sljit_u8 *new_start)
{
	sljit_uw aligned_old_start;
	sljit_uw aligned_new_start;

	if ((new_start < stack->min_start) || (new_start >= stack->end))
		return NULL;

#ifdef _WIN32
	aligned_new_start = (sljit_uw)new_start & ~sljit_page_align;
	aligned_old_start = ((sljit_uw)stack->start) & ~sljit_page_align;
	if (aligned_new_start != aligned_old_start) {
		if (aligned_new_start < aligned_old_start) {
			if (!VirtualAlloc((void*)aligned_new_start, aligned_old_start - aligned_new_start, MEM_COMMIT, PAGE_READWRITE))
				return NULL;
		}
		else {
			if (!VirtualFree((void*)aligned_old_start, aligned_new_start - aligned_old_start, MEM_DECOMMIT))
				return NULL;
		}
	}
#else
	if (stack->start < new_start) {
		aligned_new_start = (sljit_uw)new_start & ~sljit_page_align;
		aligned_old_start = ((sljit_uw)stack->start) & ~sljit_page_align;
		/* If madvise is available, we release the unnecessary space. */
#if defined(MADV_DONTNEED)
		if (aligned_new_start > aligned_old_start)
			madvise((void*)aligned_old_start, aligned_new_start - aligned_old_start, MADV_DONTNEED);
#elif defined(POSIX_MADV_DONTNEED)
		if (aligned_new_start > aligned_old_start)
			posix_madvise((void*)aligned_old_start, aligned_new_start - aligned_old_start, POSIX_MADV_DONTNEED);
#endif
	}
#endif
	stack->start = new_start;
	return new_start;
}

#endif /* SLJIT_UTIL_STACK */

#endif
