/*************************************************************************/
/*  shared_mem_access_posix.cpp                                          */
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

#include "shared_mem_access_posix.h"

#if defined(UNIX_ENABLED)

//#include "core/os/memory.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
//#include <stdio.h>
#include <fcntl.h>
//#include <sys/mman.h>
//#include <sys/stat.h>

SharedMemAccess *SharedMemAccessPosix::create_func_posix(const String &p_name) {

	return memnew(SharedMemAccessPosix(p_name));
}

void SharedMemAccessPosix::make_default() {

	create_func = create_func_posix;
}

Error SharedMemAccessPosix::open() {

	ERR_EXPLAIN("Already open.");
	ERR_FAIL_COND_V(fd != -1, ERR_ALREADY_IN_USE);

	fd = shm_open(("/" + name).utf8().get_data(), O_CREAT | O_RDWR, 0600);
	if (fd == -1) {
		ERR_EXPLAIN("Cannot create/open.");
		ERR_FAIL_V(ERR_CANT_OPEN);
	}

	return OK;
}

Error SharedMemAccessPosix::close() {

	ERR_EXPLAIN("Not open.");
	ERR_FAIL_COND_V(!is_open(), ERR_UNCONFIGURED);

	if (!lock()) {
		ERR_EXPLAIN("Cannot lock.");
		ERR_FAIL_V(ERR_BUSY);
	}

	if (is_allocator) {
		shm_unlink(("/" + name).utf8().get_data());
		is_allocator = false;
	}

	::close(fd);
	fd = -1;

	unlock();

	return OK;
}

_FORCE_INLINE_ bool SharedMemAccessPosix::is_open() {

	return fd != -1;
}

void *SharedMemAccessPosix::lock() {

	ERR_EXPLAIN("Not open.");
	ERR_FAIL_COND_V(!is_open(), NULL);

	ERR_EXPLAIN("Lock already held.");
	ERR_FAIL_COND_V(is_locking(), NULL);

	size = lseek(fd, 0, SEEK_END);

	lock_addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (lock_addr == MAP_FAILED) {
		ERR_EXPLAIN("Cannot map memory.");
		ERR_FAIL_V(NULL);
	}

	return lock_addr;
}

void SharedMemAccessPosix::unlock() {

	ERR_EXPLAIN("Lock not held.");
	ERR_FAIL_COND(!is_locking());

	munmap(lock_addr, size);
	lock_addr = NULL;
	size = 0;
}

_FORCE_INLINE_ bool SharedMemAccessPosix::is_locking() {

	return lock_addr;
}

void *SharedMemAccessPosix::set_size(uint64_t p_size) {

	ERR_EXPLAIN("Lock must be held to set the size.");
	ERR_FAIL_COND_V(!is_locking(), NULL);

	if (ftruncate(fd, p_size) == -1) {
		ERR_EXPLAIN("Cannot set new size. (Out of memory?)");
		ERR_FAIL_V(NULL);
	}

	munmap(lock_addr, size);

	// Try to remap at former address
	lock_addr = mmap(lock_addr, p_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (lock_addr == MAP_FAILED) {
		unlock();

		ERR_EXPLAIN("Cannot remap memory. Lock released.");
		ERR_FAIL_V(NULL);
	}

	return lock_addr;
}

uint64_t SharedMemAccessPosix::get_size() {

	ERR_EXPLAIN("Lock must be held to get the size.");
	ERR_FAIL_COND_V(!is_locking(), 0);

	return size;
}

SharedMemAccessPosix::SharedMemAccessPosix(const String &p_name) :
		name(p_name),
		is_allocator(false),
		size(0),
		fd(-1),
		lock_addr(NULL) {
}

SharedMemAccessPosix::~SharedMemAccessPosix() {

	if (is_open()) {

		if (is_locking()) {
			unlock();
		}

		if (close() != OK && is_allocator) {
			ERR_PRINTS("Leaking shared memory '" + name + "'.");
		}
	}
}

#endif
