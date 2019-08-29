/*************************************************************************/
/*  shared_memory_dummy.cpp                                              */
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

#if defined(NO_SHARED_MEMORY)

#include "shared_memory_dummy.h"
#include "core/os/memory.h"

SharedMemory *SharedMemoryDummy::create_func_dummy(const String &p_name) {

	return memnew(SharedMemoryDummy(p_name));
}

void SharedMemoryDummy::make_default() {

	create_func = create_func_dummy;
}

Error SharedMemoryDummy::open() {

	return ERR_UNAVAILABLE;
}

void SharedMemoryDummy::close() {}

bool SharedMemoryDummy::is_open() {

	return false;
}

uint8_t *SharedMemoryDummy::begin_access() {

	return nullptr;
}

void SharedMemoryDummy::end_access() {}

uint8_t *SharedMemoryDummy::set_size(int64_t p_size) {

	return nullptr;
}

int64_t SharedMemoryDummy::get_size() {

	return -1;
}

SharedMemoryDummy::SharedMemoryDummy(const String &p_name) {}

SharedMemoryDummy::~SharedMemoryDummy() {}

#endif
