/*************************************************************************/
/*  shared_memory_posix.cpp                                              */
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

#if defined(UNIX_ENABLED) && !defined(NO_SHARED_MEMORY)

#include "shared_memory_posix.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

SharedMemory *SharedMemoryPosix::create_func_posix(const String &p_name) {

	return memnew(SharedMemoryPosix(p_name));
}

void SharedMemoryPosix::make_default() {

	create_func = create_func_posix;
}

Error SharedMemoryPosix::open() {

	ERR_FAIL_COND_V_MSG(fd != -1, ERR_ALREADY_IN_USE, ERR_STR_ALREADY_OPEN);

	fd = shm_open(name.get_data(), O_CREAT | O_RDWR, 0600);
	if (fd == -1) {
		ERR_FAIL_V_MSG(ERR_CANT_OPEN, ERR_STR_CANNOT_CREATE_OR_OPEN);
	}

	return OK;
}

void SharedMemoryPosix::close() {

	if (!is_open()) {
		return;
	}

	if (map_addr) {
		WARN_PRINT(WARN_STR_CLOSING_BEFORE_END_ACCESS);
		end_access();
	}
	prev_map_addr = nullptr;

	::close(fd);
	fd = -1;

	shm_unlink(name.get_data());
}

_FORCE_INLINE_ bool SharedMemoryPosix::is_open() {

	return fd != -1;
}

uint8_t *SharedMemoryPosix::begin_access() {

	if (map_addr) {
		WARN_PRINT(WARN_STR_BEGIN_ACCESS_WHILE_ALREADY);
	}

	ERR_FAIL_COND_V_MSG(!is_open(), nullptr, ERR_STR_NOT_OPEN);

	int64_t size = _get_size_internal();
	if (size == -1) {
		return nullptr;
	} else if (size == 0) {
		map_addr = static_cast<uint8_t *>(UNSIZED);
		map_size = 0;
	} else {
		// Try to remap at the same address as the last time
		map_addr = (uint8_t *)mmap(prev_map_addr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		if (map_addr == MAP_FAILED) {
			map_addr = nullptr;
			map_size = 0;
		} else {
			map_size = size;
		}
	}

	return map_addr;
}

void SharedMemoryPosix::end_access() {

	if (!map_addr) {
		WARN_PRINT(WARN_STR_END_ACCESS_WITHOUT_BEGIN_ACCESS);
		return;
	}

	if (map_addr != UNSIZED) {
		munmap(map_addr, map_size);
		prev_map_addr = map_addr;
		map_size = 0;
	}
	map_addr = nullptr;
}

uint8_t *SharedMemoryPosix::set_size(int64_t p_size) {

	ERR_EXPLAIN(ERR_STR_SIZE_NOT_AVAILABLE);
	ERR_FAIL_COND_V(!map_addr, nullptr);

	end_access();

	ERR_FAIL_COND_V(p_size <= 0, nullptr);

	if (ftruncate(fd, p_size) == -1) {
		return nullptr;
	}

	return begin_access();
}

int64_t SharedMemoryPosix::get_size() {

	ERR_EXPLAIN(ERR_STR_SIZE_NOT_AVAILABLE);
	ERR_FAIL_COND_V(!map_addr, -1);

	return _get_size_internal();
}

int64_t SharedMemoryPosix::_get_size_internal() {

#if defined(__linux__)
	// Linux seems to support this (should be faster)
	return lseek(fd, 0, SEEK_END);
#else
	struct stat st;
	if (fstat(fd, &st) != -1) {
		return st.st_size;
	} else {
		return -1;
	}
#endif
}

SharedMemoryPosix::SharedMemoryPosix(const String &p_name) :
		name(("/" + p_name).ascii()),
		fd(-1),
		map_addr(nullptr),
		map_size(0),
		prev_map_addr(nullptr) {
}

SharedMemoryPosix::~SharedMemoryPosix() {

	if (is_open()) {
		close();
	}
}

#endif
