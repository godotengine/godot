/*************************************************************************/
/*  memory_pool_dynamic_prealloc.h                                       */
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
#ifndef MEMORY_POOL_DYNAMIC_PREALLOC_H
#define MEMORY_POOL_DYNAMIC_PREALLOC_H

#include "pool_allocator.h"
#include "core/os/memory_pool_dynamic.h"

class MemoryPoolDynamicPrealloc : public MemoryPoolDynamic {

	PoolAllocator *pool_alloc;

public:

	virtual ID alloc(size_t p_amount,const char* p_description);
	virtual void free(ID p_id);
	virtual Error realloc(ID p_id, size_t p_amount);
	virtual bool is_valid(ID p_id);
	virtual size_t get_size(ID p_id) const;
	virtual const char* get_description(ID p_id) const;

	virtual Error lock(ID p_id);
	virtual void * get(ID p_ID);
	virtual Error unlock(ID p_id);
	virtual bool is_locked(ID p_id) const;

	virtual size_t get_available_mem() const;
	virtual size_t get_total_usage() const;

	MemoryPoolDynamicPrealloc(void * p_mem,int p_size, int p_align = 16, int p_max_entries=PoolAllocator::DEFAULT_MAX_ALLOCS);
	~MemoryPoolDynamicPrealloc();
};

#endif // MEMORY_POOL_DYNAMIC_PREALLOC_H
