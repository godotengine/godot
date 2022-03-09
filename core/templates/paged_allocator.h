/*************************************************************************/
/*  paged_allocator.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef PAGED_ALLOCATOR_H
#define PAGED_ALLOCATOR_H

#include "core/os/memory.h"
#include "core/os/mutex.h"
#include "core/string/ustring.h"
#include "core/typedefs.h"

#include <type_traits>

// Uncomment this to replace PagedAllocator with malloc for testing
// #define GODOT_PAGED_ALLOCATOR_FALLBACK_TO_MALLOC

// Uncomment this to get printlines to monitor growth
// #define GODOT_PAGED_ALLOCATOR_REPORT_GROWTH
#ifdef GODOT_PAGED_ALLOCATOR_REPORT_GROWTH
#include "core/os/os.h"
#include "core/string/print_string.h"
#endif

template <class T, bool thread_safe = false, uint32_t DEFAULT_SIZE_UNITS = 4096>
class PagedAllocator {
	T **page_pool = nullptr;
	T ***available_pool = nullptr;
	uint32_t pages_allocated = 0;
	uint32_t allocs_available = 0;

	uint32_t page_shift = 0;
	uint32_t page_mask = 0;
	uint32_t page_size = 0;

#ifdef DEV_ENABLED
	// Keep a running count in DEV builds just for statistics.
	// No need to keep this thread safe.
	int32_t total_allocs;
#endif

	Mutex mutex;

public:
	// Only call this version if you intend to call placement new yourself.
	T *raw_alloc() {
#ifdef DEV_ENABLED
		total_allocs++;
#endif

#ifdef GODOT_PAGED_ALLOCATOR_FALLBACK_TO_MALLOC
		return (T *)malloc(sizeof(T));
#else

		if (thread_safe) {
			mutex.lock();
		}
		if (unlikely(allocs_available == 0)) {
			// This deals with global order of construction issues,
			// if a client object is constructed and calls alloc before
			// the PagedAllocator is constructed.
			if (!is_configured()) {
				WARN_PRINT_ONCE("Benign - PagedAllocator order of construction may be incorrect.");
				configure(DEFAULT_SIZE_UNITS);
			}

#ifdef GODOT_PAGED_ALLOCATOR_REPORT_GROWTH
			if (OS::get_singleton()) {
				print_line(String("PAGED_ALLOCATOR growing ") + String(typeid(T).name()));
			}
#endif
			uint32_t pages_used = pages_allocated;

			pages_allocated++;
			page_pool = (T **)memrealloc(page_pool, sizeof(T *) * pages_allocated);
			available_pool = (T ***)memrealloc(available_pool, sizeof(T **) * pages_allocated);

			page_pool[pages_used] = (T *)memalloc(sizeof(T) * page_size);
			available_pool[pages_used] = (T **)memalloc(sizeof(T *) * page_size);

			for (uint32_t i = 0; i < page_size; i++) {
				available_pool[0][i] = &page_pool[pages_used][i];
			}
			allocs_available += page_size;
		}

		allocs_available--;
		T *alloc = available_pool[allocs_available >> page_shift][allocs_available & page_mask];
		if (thread_safe) {
			mutex.unlock();
		}
		return alloc;
#endif
	}

	template <class... Args>
	T *alloc(const Args &&...p_args) {
		T *alloc = raw_alloc();
		memnew_placement(alloc, T(p_args...));
		return alloc;
	}

	void free(T *p_mem) {
#ifdef DEV_ENABLED
		total_allocs--;
#endif

#ifdef GODOT_PAGED_ALLOCATOR_FALLBACK_TO_MALLOC
		p_mem->~T();
		::free(p_mem);
		return;
#else
		if (thread_safe) {
			mutex.lock();
		}
		p_mem->~T();
		available_pool[allocs_available >> page_shift][allocs_available & page_mask] = p_mem;
		allocs_available++;
		if (thread_safe) {
			mutex.unlock();
		}
#endif
	}

	void reset(bool p_allow_unfreed = false) {
		if (!p_allow_unfreed || !std::is_trivially_destructible<T>::value) {
			ERR_FAIL_COND(allocs_available < pages_allocated * page_size);
		}
		if (pages_allocated) {
			for (uint32_t i = 0; i < pages_allocated; i++) {
				memfree(page_pool[i]);
				memfree(available_pool[i]);
			}
			memfree(page_pool);
			memfree(available_pool);
			page_pool = nullptr;
			available_pool = nullptr;
			pages_allocated = 0;
			allocs_available = 0;
		}
	}
	bool is_configured() const {
		return page_size > 0;
	}

	uint64_t estimate_memory_use() const {
		return ((uint64_t)pages_allocated * page_size * sizeof(T));
	}

	void configure(uint32_t p_page_size) {
		ERR_FAIL_COND(page_pool != nullptr); //sanity check
		ERR_FAIL_COND(p_page_size == 0);
		page_size = nearest_power_of_2_templated(p_page_size);
		page_mask = page_size - 1;
		page_shift = get_shift_from_power_of_2(page_size);
	}

#ifdef DEV_ENABLED
	int32_t get_total_allocs() const { return total_allocs; }
#endif

	PagedAllocator(uint32_t p_page_size = DEFAULT_SIZE_UNITS) { // power of 2 recommended because of alignment with OS page sizes. Even if element is bigger, its still a multiple and get rounded amount of pages
		configure(p_page_size);
	}

	~PagedAllocator() {
		if (allocs_available < pages_allocated * page_size) {
			if (Godot::g_leak_reporting_enabled) {
				ERR_FAIL_COND_MSG(allocs_available < pages_allocated * page_size, String("Pages in use exist at exit in PagedAllocator: ") + String(typeid(T).name()));
			}
			return;
		}
		reset();
	}
};

#endif // PAGED_ALLOCATOR_H
