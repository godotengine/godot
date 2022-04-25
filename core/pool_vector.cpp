/*************************************************************************/
/*  pool_vector.cpp                                                      */
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

#include "pool_vector.h"

#ifdef GODOT_POOL_VECTOR_REPORT_LEAKS
#include "core/print_string.h"
#endif

Mutex MemoryPool::counter_mutex;
uint32_t MemoryPool::allocs_used = 0;
size_t MemoryPool::total_memory = 0;
size_t MemoryPool::max_memory = 0;

#ifdef GODOT_POOL_VECTOR_REPORT_LEAKS
struct PoolVector_AllocLog {
	void *addr;
	int line;
};

LocalVector<PoolVector_AllocLog> g_pool_vector_alloc_list;

void MemoryPool::report_alloc(void *p_alloc, int p_line) {
	MemoryPool::counter_mutex.lock();
	PoolVector_AllocLog l;
	l.addr = p_alloc;
	l.line = p_line;

	g_pool_vector_alloc_list.push_back(l);
	MemoryPool::counter_mutex.unlock();
}

void MemoryPool::report_free(void *p_alloc) {
	MemoryPool::counter_mutex.lock();
	for (unsigned int n = 0; n < g_pool_vector_alloc_list.size(); n++) {
		if (g_pool_vector_alloc_list[n].addr == p_alloc) {
			g_pool_vector_alloc_list.remove_unordered(n);
			MemoryPool::counter_mutex.unlock();
			return;
		}
	}
	MemoryPool::alloc_mutex.unlock();

	ERR_PRINT("report_free alloc not found");
}

#endif // !GODOT_POOL_VECTOR_REPORT_LEAKS

void MemoryPool::report_leaks() {
#ifdef GODOT_POOL_VECTOR_REPORT_LEAKS
	print_line("MemoryPool reports " + itos(g_pool_vector_alloc_list.size()) + " leaks.");
#endif
}
