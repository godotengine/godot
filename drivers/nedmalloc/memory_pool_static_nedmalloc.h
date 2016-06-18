/*************************************************************************/
/*  memory_pool_static_nedmalloc.h                                       */
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
#ifdef NEDMALLOC_ENABLED

//
// C++ Interface: memory_static_malloc
//
// Description: 
//
//
// Author: Juan Linietsky <red@lunatea>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef MEMORY_POOL_STATIC_NEDMALLOC_H
#define MEMORY_POOL_STATIC_NEDMALLOC_H

#include "os/memory_pool_static.h"
#include "os/mutex.h"
/**
	@author Juan Linietsky <red@lunatea>
*/
class MemoryPoolStaticNedMalloc : public MemoryPoolStatic {

	Mutex *mutex;

public:

	void* alloc(size_t p_bytes,const char *p_description=""); ///< Pointer in p_description shold be to a const char const like "hello"
	void* realloc(void *p_memory,size_t p_bytes); ///< Pointer in
	void free(void *p_ptr); ///< Pointer in p_description shold be to a const char const
	virtual size_t get_available_mem() const;
	virtual size_t get_total_usage();
				
	/* Most likely available only if memory debugger was compiled in */
	virtual int get_alloc_count();
	virtual void * get_alloc_ptr(int p_alloc_idx);
	virtual const char* get_alloc_description(int p_alloc_idx);
	virtual size_t get_alloc_size(int p_alloc_idx);
	
	virtual void debug_print_all_memory();	
	
	MemoryPoolStaticNedMalloc();
	~MemoryPoolStaticNedMalloc();

};

#endif

#endif
