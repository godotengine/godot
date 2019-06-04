/*************************************************************************/
/*  memory_pool_dynamic.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef MEMORY_POOL_DYNAMIC_H
#define MEMORY_POOL_DYNAMIC_H

#include "typedefs.h"

class MemoryPoolDynamic {

	static MemoryPoolDynamic *singleton;

protected:
	friend class Memory;
	friend class MID;

	enum {

		INVALID_ID = 0xFFFFFFFF
	};

	static MemoryPoolDynamic *get_singleton();

	typedef uint64_t ID;

	virtual ID alloc(size_t p_amount, const char *p_description) = 0;
	virtual void free(ID p_id) = 0;
	virtual Error realloc(ID p_id, size_t p_amount) = 0;
	virtual bool is_valid(ID p_id) = 0;
	virtual size_t get_size(ID p_id) const = 0;
	virtual const char *get_description(ID p_id) const = 0;

	virtual Error lock(ID p_id) = 0;
	virtual void *get(ID p_ID) = 0;
	virtual Error unlock(ID p_id) = 0;
	virtual bool is_locked(ID p_id) const = 0;

	virtual size_t get_available_mem() const = 0;
	virtual size_t get_total_usage() const = 0;

	MemoryPoolDynamic();

public:
	virtual ~MemoryPoolDynamic();
};

#endif
