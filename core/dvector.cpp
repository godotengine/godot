/*************************************************************************/
/*  dvector.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "dvector.h"

Mutex *dvector_lock = NULL;

PoolAllocator *MemoryPool::memory_pool = NULL;
uint8_t *MemoryPool::pool_memory = NULL;
size_t *MemoryPool::pool_size = NULL;

MemoryPool::Alloc *MemoryPool::allocs = NULL;
MemoryPool::Alloc *MemoryPool::free_list = NULL;
uint32_t MemoryPool::alloc_count = 0;
uint32_t MemoryPool::allocs_used = 0;
Mutex *MemoryPool::alloc_mutex = NULL;

size_t MemoryPool::total_memory = 0;
size_t MemoryPool::max_memory = 0;

void MemoryPool::setup(uint32_t p_max_allocs) {

	allocs = memnew_arr(Alloc, p_max_allocs);
	alloc_count = p_max_allocs;
	allocs_used = 0;

	for (uint32_t i = 0; i < alloc_count - 1; i++) {

		allocs[i].free_list = &allocs[i + 1];
	}

	free_list = &allocs[0];

	alloc_mutex = Mutex::create();
}

void MemoryPool::cleanup() {

	memdelete_arr(allocs);
	memdelete(alloc_mutex);

	ERR_EXPLAINC("There are still MemoryPool allocs in use at exit!");
	ERR_FAIL_COND(allocs_used > 0);
}
