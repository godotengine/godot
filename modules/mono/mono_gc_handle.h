/*************************************************************************/
/*  mono_gc_handle.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef CSHARP_GC_HANDLE_H
#define CSHARP_GC_HANDLE_H

#include <mono/jit/jit.h>

#include "core/reference.h"

class MonoGCHandle : public Reference {
	GDCLASS(MonoGCHandle, Reference);

	bool released;
	bool weak;
	uint32_t handle;

public:
	enum HandleType {
		STRONG_HANDLE,
		WEAK_HANDLE
	};

	static uint32_t new_strong_handle(MonoObject *p_object);
	static uint32_t new_strong_handle_pinned(MonoObject *p_object);
	static uint32_t new_weak_handle(MonoObject *p_object);
	static void free_handle(uint32_t p_gchandle);

	static Ref<MonoGCHandle> create_strong(MonoObject *p_object);
	static Ref<MonoGCHandle> create_weak(MonoObject *p_object);

	_FORCE_INLINE_ bool is_released() { return released; }
	_FORCE_INLINE_ bool is_weak() { return weak; }

	_FORCE_INLINE_ MonoObject *get_target() const { return released ? NULL : mono_gchandle_get_target(handle); }

	_FORCE_INLINE_ void set_handle(uint32_t p_handle, HandleType p_handle_type) {
		released = false;
		weak = p_handle_type == WEAK_HANDLE;
		handle = p_handle;
	}
	void release();

	MonoGCHandle(uint32_t p_handle, HandleType p_handle_type);
	~MonoGCHandle();
};

#endif // CSHARP_GC_HANDLE_H
