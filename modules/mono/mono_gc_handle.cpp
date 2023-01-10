/**************************************************************************/
/*  mono_gc_handle.cpp                                                    */
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

#include "mono_gc_handle.h"

#include "mono_gd/gd_mono.h"

uint32_t MonoGCHandle::new_strong_handle(MonoObject *p_object) {
	return mono_gchandle_new(p_object, /* pinned: */ false);
}

uint32_t MonoGCHandle::new_strong_handle_pinned(MonoObject *p_object) {
	return mono_gchandle_new(p_object, /* pinned: */ true);
}

uint32_t MonoGCHandle::new_weak_handle(MonoObject *p_object) {
	return mono_gchandle_new_weakref(p_object, /* track_resurrection: */ false);
}

void MonoGCHandle::free_handle(uint32_t p_gchandle) {
	mono_gchandle_free(p_gchandle);
}

Ref<MonoGCHandle> MonoGCHandle::create_strong(MonoObject *p_object) {
	return memnew(MonoGCHandle(new_strong_handle(p_object), STRONG_HANDLE));
}

Ref<MonoGCHandle> MonoGCHandle::create_weak(MonoObject *p_object) {
	return memnew(MonoGCHandle(new_weak_handle(p_object), WEAK_HANDLE));
}

void MonoGCHandle::release() {
#ifdef DEBUG_ENABLED
	CRASH_COND(!released && GDMono::get_singleton() == NULL);
#endif

	if (!released && GDMono::get_singleton()->is_runtime_initialized()) {
		free_handle(handle);
		released = true;
	}
}

MonoGCHandle::MonoGCHandle(uint32_t p_handle, HandleType p_handle_type) {
	released = false;
	weak = p_handle_type == WEAK_HANDLE;
	handle = p_handle;
}

MonoGCHandle::~MonoGCHandle() {
	release();
}
