/**************************************************************************/
/*  memory.h                                                              */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef MEMORY_H
#define MEMORY_H

#include "core/error_macros.h"
#include "core/os/memory_tracker.h"
#include "core/safe_refcount.h"

#include <stddef.h>
#include <type_traits>

#ifndef PAD_ALIGN
#define PAD_ALIGN 16 //must always be greater than this at much
#endif

class Memory {
#ifdef DEBUG_ENABLED
	static SafeNumeric<uint64_t> mem_usage;
	static SafeNumeric<uint64_t> max_usage;
#endif

	static SafeNumeric<uint64_t> alloc_count;

public:
#ifdef ALLOCATION_TRACKING_ENABLED
	static void *alloc_static_tracked(size_t p_bytes, const char *p_filename, int p_line, bool p_pad_align = false);
	static void *realloc_static_tracked(void *p_memory, size_t p_bytes, const char *p_filename, int p_line, bool p_pad_align = false);
#endif
	static void *alloc_static(size_t p_bytes, bool p_pad_align = false);
	static void *realloc_static(void *p_memory, size_t p_bytes, bool p_pad_align = false);
	static void free_static(void *p_ptr, bool p_pad_align = false);

	static uint64_t get_mem_available();
	static uint64_t get_mem_usage();
	static uint64_t get_mem_max_usage();
};

class DefaultAllocator {
public:
#ifdef ALLOCATION_TRACKING_ENABLED
	// This kind of sucks because the tracking will only tell us the DefaultAllocator file, but is better than nothing...
	_FORCE_INLINE_ static void *alloc(size_t p_memory) { return Memory::alloc_static_tracked(p_memory, __FILE__, __LINE__, false); }
#else
	_FORCE_INLINE_ static void *alloc(size_t p_memory) { return Memory::alloc_static(p_memory, false); }
#endif
	_FORCE_INLINE_ static void free(void *p_ptr) { Memory::free_static(p_ptr, false); }
};

void *operator new(size_t p_size, const char *p_description); ///< operator new that takes a description and uses MemoryStaticPool
void *operator new(size_t p_size, void *(*p_allocfunc)(size_t p_size)); ///< operator new that takes a description and uses MemoryStaticPool

void *operator new(size_t p_size, void *p_pointer, size_t check, const char *p_description); ///< operator new that takes a description and uses a pointer to the preallocated memory

#ifdef _MSC_VER
// When compiling with VC++ 2017, the above declarations of placement new generate many irrelevant warnings (C4291).
// The purpose of the following definitions is to muffle these warnings, not to provide a usable implementation of placement delete.
void operator delete(void *p_mem, const char *p_description);
void operator delete(void *p_mem, void *(*p_allocfunc)(size_t p_size));
void operator delete(void *p_mem, void *p_pointer, size_t check, const char *p_description);
#endif

#ifdef ALLOCATION_TRACKING_ENABLED
#define memalloc(m_size) Memory::alloc_static_tracked(m_size, __FILE__, __LINE__)
#define memrealloc(m_mem, m_size) Memory::realloc_static_tracked(m_mem, m_size, __FILE__, __LINE__)
#else
#define memalloc(m_size) Memory::alloc_static(m_size)
#define memrealloc(m_mem, m_size) Memory::realloc_static(m_mem, m_size)
#endif
#define memfree(m_mem) Memory::free_static(m_mem)

_ALWAYS_INLINE_ void postinitialize_handler(void *) {}

template <class T>
_ALWAYS_INLINE_ T *_post_initialize(T *p_obj) {
	postinitialize_handler(p_obj);
	return p_obj;
}

#ifdef ALLOCATION_TRACKING_ENABLED

template <class T>
_ALWAYS_INLINE_ T *_post_initialize_tracked(T *p_obj, const char *p_filename, int p_line) {
	postinitialize_handler(p_obj);
	AllocationTracking::add_alloc(p_obj, sizeof(T), p_filename, p_line);
	return p_obj;
}
#define memnew(m_class) _post_initialize_tracked(new ("") m_class, __FILE__, __LINE__)
#else
#define memnew(m_class) _post_initialize(new ("") m_class)
#endif

_ALWAYS_INLINE_ void *operator new(size_t p_size, void *p_pointer, size_t check, const char *p_description) {
	//void *failptr=0;
	//ERR_FAIL_COND_V( check < p_size , failptr); /** bug, or strange compiler, most likely */

	return p_pointer;
}

#define memnew_allocator(m_class, m_allocator) _post_initialize(new (m_allocator::alloc) m_class)
#define memnew_placement(m_placement, m_class) _post_initialize(new (m_placement, sizeof(m_class), "") m_class)

_ALWAYS_INLINE_ bool predelete_handler(void *) {
	return true;
}

template <class T>
void memdelete(T *p_class) {
#ifdef ALLOCATION_TRACKING_ENABLED
	AllocationTracking::remove_alloc(p_class);
#endif
	if (!predelete_handler(p_class)) {
		return; // doesn't want to be deleted
	}
	if (!std::is_trivially_destructible<T>::value) {
		p_class->~T();
	}

	Memory::free_static(p_class, false);
}

template <class T, class A>
void memdelete_allocator(T *p_class) {
	if (!predelete_handler(p_class)) {
		return; // doesn't want to be deleted
	}
	if (!std::is_trivially_destructible<T>::value) {
		p_class->~T();
	}

	A::free(p_class);
}

#define memdelete_notnull(m_v) \
	{                          \
		if (m_v)               \
			memdelete(m_v);    \
	}

#ifdef ALLOCATION_TRACKING_ENABLED
#define memnew_arr(m_class, m_count) memnew_arr_template<m_class>(m_count, __FILE__, __LINE__)
#else
#define memnew_arr(m_class, m_count) memnew_arr_template<m_class>(m_count)
#endif

template <typename T>
#ifdef ALLOCATION_TRACKING_ENABLED
T *memnew_arr_template(size_t p_elements, const char *p_filename, int p_line, const char *p_descr = "") {
#else
T *memnew_arr_template(size_t p_elements, const char *p_descr = "") {
#endif
	if (p_elements == 0) {
		return nullptr;
	}
	/** overloading operator new[] cannot be done , because it may not return the real allocated address (it may pad the 'element count' before the actual array). Because of that, it must be done by hand. This is the
	same strategy used by std::vector, and the PoolVector class, so it should be safe.*/

	size_t len = sizeof(T) * p_elements;
#ifdef ALLOCATION_TRACKING_ENABLED
	uint64_t *mem = (uint64_t *)Memory::alloc_static_tracked(len, p_filename, p_line, true);
#else
	uint64_t *mem = (uint64_t *)Memory::alloc_static(len, true);
#endif
	T *failptr = nullptr; //get rid of a warning
	ERR_FAIL_COND_V(!mem, failptr);
	*(mem - 1) = p_elements;

	if (!std::is_trivially_constructible<T>::value) {
		T *elems = (T *)mem;

		/* call operator new */
		for (size_t i = 0; i < p_elements; i++) {
			new (&elems[i], sizeof(T), p_descr) T;
		}
	}

	return (T *)mem;
}

/**
 * Wonders of having own array functions, you can actually check the length of
 * an allocated-with memnew_arr() array
 */

template <typename T>
size_t memarr_len(const T *p_class) {
	uint64_t *ptr = (uint64_t *)p_class;
	return *(ptr - 1);
}

template <typename T>
void memdelete_arr(T *p_class) {
	uint64_t *ptr = (uint64_t *)p_class;

	if (!std::is_trivially_destructible<T>::value) {
		uint64_t elem_count = *(ptr - 1);

		for (uint64_t i = 0; i < elem_count; i++) {
			p_class[i].~T();
		}
	}

	Memory::free_static(ptr, true);
}

struct _GlobalNil {
	int color;
	_GlobalNil *right;
	_GlobalNil *left;
	_GlobalNil *parent;

	_GlobalNil();
};

struct _GlobalNilClass {
	static _GlobalNil _nil;
};

#endif // MEMORY_H
