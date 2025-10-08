/**************************************************************************/
/*  dr_bridge.h                                                           */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "core/io/file_access.h"
#include "core/os/memory.h"
#include "core/typedefs.h"

typedef struct {
	void *user_data;
	void *(*malloc_func)(size_t size, void *user_data);
	void *(*realloc_func)(void *ptr, size_t size, void *user_data);
	void (*free_func)(void *ptr, void *user_data);
} dr_allocation_callbacks;

static void *dr_memalloc(size_t size, void *user_data) {
	return memalloc(size);
}

static void *dr_memrealloc(void *ptr, size_t size, void *user_data) {
	return memrealloc(ptr, size);
}

static void dr_memfree(void *ptr, void *user_data) {
	if (ptr) {
		memfree(ptr);
	}
}

const dr_allocation_callbacks dr_alloc_calls = {
	nullptr,
	dr_memalloc,
	dr_memrealloc,
	dr_memfree
};
