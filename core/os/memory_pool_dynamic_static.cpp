/*************************************************************************/
/*  memory_pool_dynamic_static.cpp                                       */
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
#include "memory_pool_dynamic_static.h"
#include "os/memory.h"
#include "os/os.h"
#include "ustring.h"
#include "print_string.h"
#include <stdio.h>

MemoryPoolDynamicStatic::Chunk *MemoryPoolDynamicStatic::get_chunk(ID p_id) {

	uint64_t check = p_id/MAX_CHUNKS;
	uint64_t idx = p_id%MAX_CHUNKS;

	if (!chunk[idx].mem || chunk[idx].check!=check)
		return NULL;

	return &chunk[idx];
}


const MemoryPoolDynamicStatic::Chunk *MemoryPoolDynamicStatic::get_chunk(ID p_id) const {

	uint64_t check = p_id/MAX_CHUNKS;
	uint64_t idx = p_id%MAX_CHUNKS;

	if (!chunk[idx].mem || chunk[idx].check!=check)
		return NULL;

	return &chunk[idx];
}

MemoryPoolDynamic::ID MemoryPoolDynamicStatic::alloc(size_t p_amount,const char* p_description) {

	_THREAD_SAFE_METHOD_

	int idx=-1;

	for (int i=0;i<MAX_CHUNKS;i++) {

		last_alloc++;
		if (last_alloc>=MAX_CHUNKS)
			last_alloc=0;

		if ( !chunk[last_alloc].mem ) {

			idx=last_alloc;
			break;
		}
	}


	if (idx==-1) {
		ERR_EXPLAIN("Out of dynamic Memory IDs");
		ERR_FAIL_V(INVALID_ID);
		//return INVALID_ID;
	}

	//chunk[idx].mem = Memory::alloc_static(p_amount,p_description);
	chunk[idx].mem = memalloc(p_amount);
	if (!chunk[idx].mem)
		return INVALID_ID;

	chunk[idx].size=p_amount;
	chunk[idx].check=++last_check;
	chunk[idx].descr=p_description;
	chunk[idx].lock=0;

	total_usage+=p_amount;
	if (total_usage>max_usage)
		max_usage=total_usage;

	ID id = chunk[idx].check*MAX_CHUNKS + (uint64_t)idx;

	return id;

}
void MemoryPoolDynamicStatic::free(ID p_id) {

	_THREAD_SAFE_METHOD_

	Chunk *c = get_chunk(p_id);
	ERR_FAIL_COND(!c);


	total_usage-=c->size;
	memfree(c->mem);

	c->mem=0;

	if (c->lock>0) {

		ERR_PRINT("Freed ID Still locked");
	}
}

Error MemoryPoolDynamicStatic::realloc(ID p_id, size_t p_amount) {

	_THREAD_SAFE_METHOD_

	Chunk *c = get_chunk(p_id);
	ERR_FAIL_COND_V(!c,ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(c->lock > 0 , ERR_LOCKED );


	void * new_mem = memrealloc(c->mem,p_amount);

	ERR_FAIL_COND_V(!new_mem,ERR_OUT_OF_MEMORY);
	total_usage-=c->size;
	c->mem=new_mem;
	c->size=p_amount;
	total_usage+=c->size;
	if (total_usage>max_usage)
		max_usage=total_usage;


	return OK;
}
bool MemoryPoolDynamicStatic::is_valid(ID p_id) {

	_THREAD_SAFE_METHOD_

	Chunk *c = get_chunk(p_id);

	return c!=NULL;

}
size_t MemoryPoolDynamicStatic::get_size(ID p_id) const {

	_THREAD_SAFE_METHOD_

	const Chunk *c = get_chunk(p_id);
	ERR_FAIL_COND_V(!c,0);

	return c->size;


}
const char* MemoryPoolDynamicStatic::get_description(ID p_id) const {

	_THREAD_SAFE_METHOD_

	const Chunk *c = get_chunk(p_id);
	ERR_FAIL_COND_V(!c,"");

	return c->descr;

}


bool MemoryPoolDynamicStatic::is_locked(ID p_id) const {

	_THREAD_SAFE_METHOD_

	const Chunk *c = get_chunk(p_id);
	ERR_FAIL_COND_V(!c,false);

	return c->lock>0;

}

Error MemoryPoolDynamicStatic::lock(ID p_id) {

	_THREAD_SAFE_METHOD_

	Chunk *c = get_chunk(p_id);
	ERR_FAIL_COND_V(!c,ERR_INVALID_PARAMETER);

	c->lock++;

	return OK;
}
void * MemoryPoolDynamicStatic::get(ID p_id) {

	_THREAD_SAFE_METHOD_

	const Chunk *c = get_chunk(p_id);
	ERR_FAIL_COND_V(!c,NULL);
	ERR_FAIL_COND_V( c->lock==0, NULL );

	return c->mem;
}
Error MemoryPoolDynamicStatic::unlock(ID p_id) {

	_THREAD_SAFE_METHOD_

	Chunk *c = get_chunk(p_id);
	ERR_FAIL_COND_V(!c,ERR_INVALID_PARAMETER);

	ERR_FAIL_COND_V( c->lock<=0, ERR_INVALID_PARAMETER );
	c->lock--;

	return OK;
}

size_t MemoryPoolDynamicStatic::get_available_mem() const {

	return Memory::get_static_mem_available();
}

size_t MemoryPoolDynamicStatic::get_total_usage() const {
	_THREAD_SAFE_METHOD_

	return total_usage;
}

MemoryPoolDynamicStatic::MemoryPoolDynamicStatic() {

	last_check=1;
	last_alloc=0;
	total_usage=0;
	max_usage=0;
}

MemoryPoolDynamicStatic::~MemoryPoolDynamicStatic() {

#ifdef DEBUG_MEMORY_ENABLED

	if (OS::get_singleton()->is_stdout_verbose()) {

		if (total_usage>0) {

			ERR_PRINT("DYNAMIC ALLOC: ** MEMORY LEAKS DETECTED **");
			ERR_PRINT(String("DYNAMIC ALLOC: "+String::num(total_usage)+" bytes of memory in use at exit.").ascii().get_data());

			ERR_PRINT("DYNAMIC ALLOC: Following is the list of leaked allocations:");

			for (int i=0;i<MAX_CHUNKS;i++) {

				if (chunk[i].mem) {

					ERR_PRINT(String("\t"+String::num(chunk[i].size)+" bytes - "+String(chunk[i].descr)).ascii().get_data());
				}
			}

			ERR_PRINT("DYNAMIC ALLOC: End of Report.");

			print_line("INFO: dynmem - max: "+itos(max_usage)+", "+itos(total_usage)+" leaked.");
		} else {

			print_line("INFO: dynmem - max: "+itos(max_usage)+", no leaks.");
		}
	}

#endif
}
