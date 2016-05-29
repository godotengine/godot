/*************************************************************************/
/*  memory_pool_dynamic_static.h                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef MEMORY_POOL_DYNAMIC_STATIC_H
#define MEMORY_POOL_DYNAMIC_STATIC_H

#include "os/memory_pool_dynamic.h"
#include "typedefs.h"
#include "os/thread_safe.h"

class MemoryPoolDynamicStatic : public MemoryPoolDynamic {

	_THREAD_SAFE_CLASS_

	enum {
		MAX_CHUNKS=65536
	};


	struct Chunk {

		uint64_t lock;
		uint64_t check;
		void *mem;
		size_t size;
		const char *descr;

		Chunk() { mem=NULL; lock=0; check=0; }
	};

	Chunk chunk[MAX_CHUNKS];
	uint64_t last_check;
	int last_alloc;
	size_t total_usage;
	size_t max_usage;

	Chunk *get_chunk(ID p_id);
	const Chunk *get_chunk(ID p_id) const;
public:

	virtual ID alloc(size_t p_amount,const char* p_description);
	virtual void free(ID p_id);
	virtual Error realloc(ID p_id, size_t p_amount);
	virtual bool is_valid(ID p_id);
	virtual size_t get_size(ID p_id) const;
	virtual const char* get_description(ID p_id) const;

	virtual bool is_locked(ID p_id) const;
	virtual Error lock(ID p_id);
	virtual void * get(ID p_ID);
	virtual Error unlock(ID p_id);

	virtual size_t get_available_mem() const;
	virtual size_t get_total_usage() const;

	MemoryPoolDynamicStatic();
	virtual ~MemoryPoolDynamicStatic();

};

#endif
