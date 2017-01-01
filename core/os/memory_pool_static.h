/*************************************************************************/
/*  memory_pool_static.h                                                 */
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
#ifndef MEMORY_POOL_STATIC_H
#define MEMORY_POOL_STATIC_H

#include <stddef.h>

#include "core/typedefs.h"

/**
	@author Juan Linietsky <red@lunatea>
*/
class MemoryPoolStatic {
private:

	static MemoryPoolStatic *singleton;

public:

	static MemoryPoolStatic *get_singleton();

	virtual void* alloc(size_t p_bytes,const char *p_description)=0; ///< Pointer in p_description shold be to a const char const like "hello"
	virtual void* realloc(void * p_memory,size_t p_bytes)=0; ///< Pointer in p_description shold be to a const char const like "hello"
	virtual void free(void *p_ptr)=0; ///< Pointer in p_description shold be to a const char const

	virtual size_t get_available_mem() const=0;
	virtual size_t get_total_usage()=0;
	virtual size_t get_max_usage()=0;

	/* Most likely available only if memory debugger was compiled in */
	virtual int get_alloc_count()=0;
	virtual void * get_alloc_ptr(int p_alloc_idx)=0;
	virtual const char* get_alloc_description(int p_alloc_idx)=0;
	virtual size_t get_alloc_size(int p_alloc_idx)=0;

	virtual void dump_mem_to_file(const char* p_file)=0;

	MemoryPoolStatic();
	virtual ~MemoryPoolStatic();

};

#endif
