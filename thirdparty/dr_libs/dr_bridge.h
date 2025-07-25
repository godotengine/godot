/**************************************************************************/
/*  dr_alloc_calls.h                                                      */
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

static size_t dr_read_fileaccess(void *p_user_data, void *p_out, size_t p_to_read) {
	Ref<FileAccess> lfile = *(Ref<FileAccess> *)p_user_data;
	return lfile->get_buffer((uint8_t *)p_out, p_to_read);
}

template <typename T>
static uint32_t dr_seek_fileaccess(void *p_user_data, int p_offset, T p_origin) {
	Ref<FileAccess> lfile = *(Ref<FileAccess> *)p_user_data;
	uint64_t new_offset;
	switch (p_origin) {
		case 0: // SEEK_SET
			new_offset = p_offset;
			break;
		case 1: // SEEK_CUR
			new_offset = lfile->get_position() + p_offset;
			break;
		case 2: // SEEK_END
			new_offset = lfile->get_length() + p_offset;
			break;
		default:
			ERR_FAIL_V_MSG(0, "Invalid seek origin in dr_seek_fileaccess.");
	}

	if (new_offset > lfile->get_length()) {
		return 0;
	}

	lfile->seek(new_offset);
	return 1;
}

static uint32_t dr_tell_fileaccess(void *p_user_data, uint64_t *p_cursor) {
	Ref<FileAccess> lfile = *(Ref<FileAccess> *)p_user_data;
	int64_t result;
	if (p_user_data == nullptr || p_cursor == nullptr) {
		return 0;
	}
	result = lfile->get_position();
	*p_cursor = result;
	return 1;
}

const dr_allocation_callbacks dr_alloc_calls = {
	nullptr,
	dr_memalloc,
	dr_memrealloc,
	dr_memfree
};
