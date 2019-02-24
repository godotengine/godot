/*************************************************************************/
/*  shared_mem_access_posix.h                                            */
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

#ifndef SHARED_MEM_ACCESS_POSIX_H
#define SHARED_MEM_ACCESS_POSIX_H

#include "core/os/shared_mem_access.h"
#include "core/ustring.h"

#if defined(UNIX_ENABLED)

class SharedMemAccessPosix : public SharedMemAccess {

	String name;
	bool is_allocator;
	uint64_t size;
	int fd;
	void *lock_addr;

	static SharedMemAccess *create_func_posix(const String &p_name);

public:
	static void make_default();

	virtual Error open();
	virtual Error close();
	virtual bool is_open();

	virtual void *lock();
	virtual void unlock();
	virtual bool is_locking();

	virtual void *set_size(uint64_t p_size);
	virtual uint64_t get_size();

	SharedMemAccessPosix(const String &p_name);
	virtual ~SharedMemAccessPosix();
};

#endif
#endif
