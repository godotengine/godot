/*************************************************************************/
/*  memory_pool_dynamic_prealloc.cpp                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "memory_pool_dynamic_prealloc.h"
#include "os/memory.h"

#include "print_string.h"
MemoryPoolDynamicPrealloc::ID MemoryPoolDynamicPrealloc::alloc(size_t p_amount,const char* p_description) {


//	print_line("dynpool - allocating: "+itos(p_amount));
	ID id = pool_alloc->alloc(p_amount);
//	print_line("dynpool - free: "+itos(pool_alloc->get_free_mem()));
	return id;

}

void MemoryPoolDynamicPrealloc::free(ID p_id)  {

	pool_alloc->free(p_id);
}

Error MemoryPoolDynamicPrealloc::realloc(ID p_id, size_t p_amount) {

	return pool_alloc->resize(p_id,p_amount);
}

bool MemoryPoolDynamicPrealloc::is_valid(ID p_id) {

	return true;
}

size_t MemoryPoolDynamicPrealloc::get_size(ID p_id) const {

	return pool_alloc->get_size(p_id);
}

const char* MemoryPoolDynamicPrealloc::get_description(ID p_id) const {

	return "";
}

Error MemoryPoolDynamicPrealloc::lock(ID p_id) {

//	print_line("lock: "+itos(p_id));
	return pool_alloc->lock(p_id);
}

void * MemoryPoolDynamicPrealloc::get(ID p_ID) {

//	print_line("get: "+itos(p_ID));
	return pool_alloc->get(p_ID);
}

Error MemoryPoolDynamicPrealloc::unlock(ID p_id) {

//	print_line("unlock: "+itos(p_id));
	pool_alloc->unlock(p_id);
	return OK;
}

bool MemoryPoolDynamicPrealloc::is_locked(ID p_id) const {

	return pool_alloc->is_locked(p_id);
}


size_t MemoryPoolDynamicPrealloc::get_available_mem() const {

	return pool_alloc->get_free_mem();
}

size_t MemoryPoolDynamicPrealloc::get_total_usage() const {

	return pool_alloc->get_used_mem();
}



MemoryPoolDynamicPrealloc::MemoryPoolDynamicPrealloc(void * p_mem,int p_size, int p_align, int p_max_entries) {

	pool_alloc = memnew( PoolAllocator(p_mem,p_size,p_align,true,p_max_entries));

}

MemoryPoolDynamicPrealloc::~MemoryPoolDynamicPrealloc() {


	memdelete( pool_alloc );
}

