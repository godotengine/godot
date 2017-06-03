/*************************************************************************/
/*  memory.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include <stdlib.h>

void *operator new(size_t p_size, const char *p_description) {

	return Memory::alloc_static(p_size, false);
}

void *operator new(size_t p_size, void *(*p_allocfunc)(size_t p_size)) {

	return p_allocfunc(p_size);
}

#include <stdio.h>

#ifdef DEBUG_ENABLED
size_t Memory::mem_usage = 0;
size_t Memory::max_usage = 0;
#endif

size_t Memory::alloc_count = 0;

void *Memory::alloc_static(size_t p_bytes, bool p_pad_align) {

#ifdef DEBUG_ENABLED
	bool prepad = true;
#else
	bool prepad = p_pad_align;
#endif

	void *mem = malloc(p_bytes + (prepad ? PAD_ALIGN : 0));

	alloc_count++;

	ERR_FAIL_COND_V(!mem, NULL);

	if (prepad) {
		uint64_t *s = (uint64_t *)mem;
		*s = p_bytes;

		uint8_t *s8 = (uint8_t *)mem;

#ifdef DEBUG_ENABLED
		mem_usage += p_bytes;
		if (mem_usage > max_usage) {
			max_usage = mem_usage;
		}
#endif
		return s8 + PAD_ALIGN;
	} else {
		return mem;
	}
}

void *Memory::realloc_static(void *p_memory, size_t p_bytes, bool p_pad_align) {

	if (p_memory == NULL) {
		return alloc_static(p_bytes, p_pad_align);
	}

	uint8_t *mem = (uint8_t *)p_memory;

#ifdef DEBUG_ENABLED
	bool prepad = true;
#else
	bool prepad = p_pad_align;
#endif

	if (prepad) {
		mem -= PAD_ALIGN;
		uint64_t *s = (uint64_t *)mem;

#ifdef DEBUG_ENABLED
		mem_usage -= *s;
		mem_usage += p_bytes;
#endif

		if (p_bytes == 0) {
			free(mem);
			return NULL;
		} else {
			*s = p_bytes;

			mem = (uint8_t *)realloc(mem, p_bytes + PAD_ALIGN);
			ERR_FAIL_COND_V(!mem, NULL);

			s = (uint64_t *)mem;

			*s = p_bytes;

			return mem + PAD_ALIGN;
		}
	} else {

		mem = (uint8_t *)realloc(mem, p_bytes);

		ERR_FAIL_COND_V(mem == NULL && p_bytes > 0, NULL);

		return mem;
	}
}

void Memory::free_static(void *p_ptr, bool p_pad_align) {

	ERR_FAIL_COND(p_ptr == NULL);

	uint8_t *mem = (uint8_t *)p_ptr;

#ifdef DEBUG_ENABLED
	bool prepad = true;
#else
	bool prepad = p_pad_align;
#endif

	alloc_count--;

	if (prepad) {
		mem -= PAD_ALIGN;
		uint64_t *s = (uint64_t *)mem;

#ifdef DEBUG_ENABLED
		mem_usage -= *s;
#endif

		free(mem);
	} else {

		free(mem);
	}
}

size_t Memory::get_mem_available() {

	return 0xFFFFFFFFFFFFF;
}

size_t Memory::get_mem_usage() {
#ifdef DEBUG_ENABLED
	return mem_usage;
#else
	return 0;
#endif
}
size_t Memory::get_mem_max_usage() {
#ifdef DEBUG_ENABLED
	return max_usage;
#else
	return 0;
#endif
}

_GlobalNil::_GlobalNil() {

	color = 1;
	left = this;
	right = this;
	parent = this;
}

_GlobalNil _GlobalNilClass::_nil;
