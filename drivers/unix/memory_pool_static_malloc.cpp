/*************************************************************************/
/*  memory_pool_static_malloc.cpp                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "memory_pool_static_malloc.h"
#include "error_macros.h"
#include "os/copymem.h"
#include "os/memory.h"
#include "os/os.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * NOTE NOTE NOTE NOTE
 * in debug mode, this prepends the memory size to the allocated block
 * so BE CAREFUL!
 */

void *MemoryPoolStaticMalloc::alloc(size_t p_bytes, const char *p_description) {

#if DEFAULT_ALIGNMENT == 1

	return _alloc(p_bytes, p_description);

#else

	size_t total;
#if defined(_add_overflow)
	if (_add_overflow(p_bytes, DEFAULT_ALIGNMENT, &total)) return NULL;
#else
	total = p_bytes + DEFAULT_ALIGNMENT;
#endif
	uint8_t *ptr = (uint8_t *)_alloc(total, p_description);
	ERR_FAIL_COND_V(!ptr, ptr);
	int ofs = (DEFAULT_ALIGNMENT - ((uintptr_t)ptr & (DEFAULT_ALIGNMENT - 1)));
	ptr[ofs - 1] = ofs;
	return (void *)(ptr + ofs);
#endif
};

void *MemoryPoolStaticMalloc::_alloc(size_t p_bytes, const char *p_description) {

	ERR_FAIL_COND_V(p_bytes == 0, 0);

#ifdef DEBUG_MEMORY_ENABLED

	MutexLock lock(mutex);

	size_t total;
#if defined(_add_overflow)
	if (_add_overflow(p_bytes, sizeof(RingPtr), &total)) return NULL;
#else
	total = p_bytes + sizeof(RingPtr);
#endif
	void *mem = malloc(total); /// add for size and ringlist

	if (!mem) {
		printf("**ERROR: out of memory while allocating %lu bytes by %s?\n", (unsigned long)p_bytes, p_description);
		printf("**ERROR: memory usage is %lu\n", (unsigned long)get_total_usage());
	};

	ERR_FAIL_COND_V(!mem, 0); //out of memory, or unreasonable request

	/* setup the ringlist element */

	RingPtr *ringptr = (RingPtr *)mem;

	/* setup the ringlist element data (description and size ) */

	ringptr->size = p_bytes;
	ringptr->descr = p_description;

	if (ringlist) { /* existing ringlist */

		/* assign next */
		ringptr->next = ringlist->next;
		ringlist->next = ringptr;
		/* assign prev */
		ringptr->prev = ringlist;
		ringptr->next->prev = ringptr;
	} else { /* non existing ringlist */

		ringptr->next = ringptr;
		ringptr->prev = ringptr;
		ringlist = ringptr;
	}

	total_mem += p_bytes;

	/* update statistics */
	if (total_mem > max_mem)
		max_mem = total_mem;

	total_pointers++;

	if (total_pointers > max_pointers)
		max_pointers = total_pointers;

	return ringptr + 1; /* return memory after ringptr */

#else
	void *mem = malloc(p_bytes);

	ERR_FAIL_COND_V(!mem, 0); //out of memory, or unreasonable request
	return mem;
#endif
}

void *MemoryPoolStaticMalloc::realloc(void *p_memory, size_t p_bytes) {

#if DEFAULT_ALIGNMENT == 1

	return _realloc(p_memory, p_bytes);
#else
	if (!p_memory)
		return alloc(p_bytes);

	size_t total;
#if defined(_add_overflow)
	if (_add_overflow(p_bytes, DEFAULT_ALIGNMENT, &total)) return NULL;
#else
	total = p_bytes + DEFAULT_ALIGNMENT;
#endif
	uint8_t *mem = (uint8_t *)p_memory;
	int ofs = *(mem - 1);
	mem = mem - ofs;
	uint8_t *ptr = (uint8_t *)_realloc(mem, total);
	ERR_FAIL_COND_V(ptr == NULL, NULL);
	int new_ofs = (DEFAULT_ALIGNMENT - ((uintptr_t)ptr & (DEFAULT_ALIGNMENT - 1)));
	if (new_ofs != ofs) {

		//printf("realloc moving %i bytes\n", p_bytes);
		movemem((ptr + new_ofs), (ptr + ofs), p_bytes);
		ptr[new_ofs - 1] = new_ofs;
	};
	return ptr + new_ofs;
#endif
};

void *MemoryPoolStaticMalloc::_realloc(void *p_memory, size_t p_bytes) {

	if (p_memory == NULL) {

		return alloc(p_bytes);
	}

	if (p_bytes == 0) {

		this->free(p_memory);
		ERR_FAIL_COND_V(p_bytes < 0, NULL);
		return NULL;
	}

#ifdef DEBUG_MEMORY_ENABLED

	MutexLock lock(mutex);

	RingPtr *ringptr = (RingPtr *)p_memory;
	ringptr--; /* go back an element to find the tingptr */

	bool single_element = (ringptr->next == ringptr) && (ringptr->prev == ringptr);
	bool is_list = (ringlist == ringptr);

	RingPtr *new_ringptr = (RingPtr *)::realloc(ringptr, p_bytes + sizeof(RingPtr));

	ERR_FAIL_COND_V(new_ringptr == 0, NULL); /// reallocation failed

	/* actualize mem used */
	total_mem -= new_ringptr->size;
	new_ringptr->size = p_bytes;
	total_mem += new_ringptr->size;

	if (total_mem > max_mem) //update statistics
		max_mem = total_mem;

	if (new_ringptr == ringptr)
		return ringptr + 1; // block didn't move, don't do anything

	if (single_element) {

		new_ringptr->next = new_ringptr;
		new_ringptr->prev = new_ringptr;
	} else {

		new_ringptr->next->prev = new_ringptr;
		new_ringptr->prev->next = new_ringptr;
	}

	if (is_list)
		ringlist = new_ringptr;

	return new_ringptr + 1;

#else
	return ::realloc(p_memory, p_bytes);
#endif
}

void MemoryPoolStaticMalloc::free(void *p_ptr) {

	ERR_FAIL_COND(!MemoryPoolStatic::get_singleton());

#if DEFAULT_ALIGNMENT == 1

	_free(p_ptr);
#else

	uint8_t *mem = (uint8_t *)p_ptr;
	int ofs = *(mem - 1);
	mem = mem - ofs;

	_free(mem);
#endif
};

void MemoryPoolStaticMalloc::_free(void *p_ptr) {

#ifdef DEBUG_MEMORY_ENABLED

	MutexLock lock(mutex);

	if (p_ptr == 0) {
		printf("**ERROR: STATIC ALLOC: Attempted free of NULL pointer.\n");
		return;
	};

	RingPtr *ringptr = (RingPtr *)p_ptr;

	ringptr--; /* go back an element to find the ringptr */

#if 0	
	{ // check for existing memory on free.
		RingPtr *p = ringlist;
		
		bool found=false;
		
		if (ringlist) {
			do {
				if (p==ringptr) {
					found=true;
					break;
				}
						
				p=p->next;
			} while (p!=ringlist);
		}
		
		if (!found) {
			printf("**ERROR: STATIC ALLOC: Attempted free of unknown pointer at %p\n",(ringptr+1));
			return;
		}
		
	}
#endif
	/* proceed to erase */

	bool single_element = (ringptr->next == ringptr) && (ringptr->prev == ringptr);
	bool is_list = (ringlist == ringptr);

	if (single_element) {
		/* just get rid of it */
		ringlist = 0;

	} else {
		/* auto-remove from ringlist */
		if (is_list)
			ringlist = ringptr->next;

		ringptr->prev->next = ringptr->next;
		ringptr->next->prev = ringptr->prev;
	}

	total_mem -= ringptr->size;
	total_pointers--;
	// catch more errors
	memset(ringptr, 0xEA, sizeof(RingPtr) + ringptr->size);
	::free(ringptr); //just free that pointer

#else
	ERR_FAIL_COND(p_ptr == 0);

	::free(p_ptr);
#endif
}

size_t MemoryPoolStaticMalloc::get_available_mem() const {

	return 0xffffffff;
}

size_t MemoryPoolStaticMalloc::get_total_usage() {

#ifdef DEBUG_MEMORY_ENABLED

	return total_mem;
#else
	return 0;
#endif
}

size_t MemoryPoolStaticMalloc::get_max_usage() {

	return max_mem;
}

/* Most likely available only if memory debugger was compiled in */
int MemoryPoolStaticMalloc::get_alloc_count() {

	return total_pointers;
}
void *MemoryPoolStaticMalloc::get_alloc_ptr(int p_alloc_idx) {

	return 0;
}
const char *MemoryPoolStaticMalloc::get_alloc_description(int p_alloc_idx) {

	return "";
}
size_t MemoryPoolStaticMalloc::get_alloc_size(int p_alloc_idx) {

	return 0;
}

void MemoryPoolStaticMalloc::dump_mem_to_file(const char *p_file) {

#ifdef DEBUG_MEMORY_ENABLED

	ERR_FAIL_COND(!ringlist); /** WTF BUG !? */
	RingPtr *p = ringlist;
	FILE *f = fopen(p_file, "wb");

	do {
		fprintf(f, "%p-%i-%s\n", p + 1, (int)p->size, (p->descr ? p->descr : ""));
		p = p->next;
	} while (p != ringlist);

	fclose(f);
#endif
}

MemoryPoolStaticMalloc::MemoryPoolStaticMalloc() {

#ifdef DEBUG_MEMORY_ENABLED
	total_mem = 0;
	total_pointers = 0;
	ringlist = 0;
	max_mem = 0;
	max_pointers = 0;

#endif

	mutex = NULL;
#ifndef NO_THREADS

	mutex = Mutex::create(); // at this point, this should work
#endif
}

MemoryPoolStaticMalloc::~MemoryPoolStaticMalloc() {

	Mutex *old_mutex = mutex;
	mutex = NULL;
	if (old_mutex)
		memdelete(old_mutex);

#ifdef DEBUG_MEMORY_ENABLED

	if (OS::get_singleton()->is_stdout_verbose()) {
		if (total_mem > 0) {
			printf("**ERROR: STATIC ALLOC: ** MEMORY LEAKS DETECTED **\n");
			printf("**ERROR: STATIC ALLOC: %i bytes of memory in use at exit.\n", (int)total_mem);

			if (1) {
				printf("**ERROR: STATIC ALLOC: Following is the list of leaked allocations: \n");

				ERR_FAIL_COND(!ringlist); /** WTF BUG !? */
				RingPtr *p = ringlist;

				do {
					printf("\t%p - %i bytes - %s\n", (RingPtr *)(p + 1), (int)p->size, (p->descr ? p->descr : ""));
					p = p->next;
				} while (p != ringlist);

				printf("**ERROR: STATIC ALLOC: End of Report.\n");
			};

			printf("mem - max %i, pointers %i, leaks %i.\n", (int)max_mem, max_pointers, (int)total_mem);
		} else {

			printf("INFO: mem - max %i, pointers %i, no leaks.\n", (int)max_mem, max_pointers);
		}
	}

#endif
}
