/*************************************************************************/
/*  mono_gc_handle.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef CSHARP_GC_HANDLE_H
#define CSHARP_GC_HANDLE_H

#include <mono/jit/jit.h>

#include "reference.h"

class MonoGCHandle : public Reference {

	GDCLASS(MonoGCHandle, Reference)

	bool released;
	uint32_t handle;

public:
	static uint32_t make_strong_handle(MonoObject *p_object);
	static uint32_t make_weak_handle(MonoObject *p_object);

	static Ref<MonoGCHandle> create_strong(MonoObject *p_object);
	static Ref<MonoGCHandle> create_weak(MonoObject *p_object);

	_FORCE_INLINE_ MonoObject *get_target() const { return released ? NULL : mono_gchandle_get_target(handle); }

	_FORCE_INLINE_ void set_handle(uint32_t p_handle) {
		handle = p_handle;
		released = false;
	}
	void release();

	MonoGCHandle(uint32_t p_handle);
	~MonoGCHandle();
};

#endif // CSHARP_GC_HANDLE_H
