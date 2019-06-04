/*************************************************************************/
/*  memory.cpp                                                           */
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
#include "memory.h"
#include "copymem.h"
#include "error_macros.h"
#include <stdio.h>
void *operator new(size_t p_size, const char *p_description) {

	return Memory::alloc_static(p_size, p_description);
}

void *operator new(size_t p_size, void *(*p_allocfunc)(size_t p_size)) {

	return p_allocfunc(p_size);
}

#include <stdio.h>

void *Memory::alloc_static(size_t p_bytes, const char *p_alloc_from) {

	ERR_FAIL_COND_V(!MemoryPoolStatic::get_singleton(), NULL);
	return MemoryPoolStatic::get_singleton()->alloc(p_bytes, p_alloc_from);
}
void *Memory::realloc_static(void *p_memory, size_t p_bytes) {

	ERR_FAIL_COND_V(!MemoryPoolStatic::get_singleton(), NULL);
	return MemoryPoolStatic::get_singleton()->realloc(p_memory, p_bytes);
}

void Memory::free_static(void *p_ptr) {

	ERR_FAIL_COND(!MemoryPoolStatic::get_singleton());
	MemoryPoolStatic::get_singleton()->free(p_ptr);
}

size_t Memory::get_static_mem_available() {

	ERR_FAIL_COND_V(!MemoryPoolStatic::get_singleton(), 0);
	return MemoryPoolStatic::get_singleton()->get_available_mem();
}

size_t Memory::get_static_mem_max_usage() {

	ERR_FAIL_COND_V(!MemoryPoolStatic::get_singleton(), 0);
	return MemoryPoolStatic::get_singleton()->get_max_usage();
}

size_t Memory::get_static_mem_usage() {

	ERR_FAIL_COND_V(!MemoryPoolStatic::get_singleton(), 0);
	return MemoryPoolStatic::get_singleton()->get_total_usage();
}

void Memory::dump_static_mem_to_file(const char *p_file) {

	MemoryPoolStatic::get_singleton()->dump_mem_to_file(p_file);
}

MID Memory::alloc_dynamic(size_t p_bytes, const char *p_descr) {

	MemoryPoolDynamic::ID id = MemoryPoolDynamic::get_singleton()->alloc(p_bytes, p_descr);

	return MID(id);
}
Error Memory::realloc_dynamic(MID p_mid, size_t p_bytes) {

	MemoryPoolDynamic::ID id = p_mid.data ? p_mid.data->id : MemoryPoolDynamic::INVALID_ID;

	if (id == MemoryPoolDynamic::INVALID_ID)
		return ERR_INVALID_PARAMETER;

	return MemoryPoolDynamic::get_singleton()->realloc(p_mid, p_bytes);
}

size_t Memory::get_dynamic_mem_available() {

	return MemoryPoolDynamic::get_singleton()->get_available_mem();
}

size_t Memory::get_dynamic_mem_usage() {

	return MemoryPoolDynamic::get_singleton()->get_total_usage();
}

_GlobalNil::_GlobalNil() {

	color = 1;
	left = this;
	right = this;
	parent = this;
}

_GlobalNil _GlobalNilClass::_nil;
