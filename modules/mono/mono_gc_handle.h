/*************************************************************************/
/*  mono_gc_handle.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/object/reference.h"

namespace gdmono {

enum class GCHandleType : char {
	NIL,
	STRONG_HANDLE,
	WEAK_HANDLE
};
}

// Manual release of the GC handle must be done when using this struct
struct MonoGCHandleData {
	uint32_t handle = 0;
	gdmono::GCHandleType type = gdmono::GCHandleType::NIL;

	_FORCE_INLINE_ bool is_released() const { return !handle; }
	_FORCE_INLINE_ bool is_weak() const { return type == gdmono::GCHandleType::WEAK_HANDLE; }

	_FORCE_INLINE_ MonoObject *get_target() const { return handle ? mono_gchandle_get_target(handle) : nullptr; }

	void release();

	MonoGCHandleData &operator=(const MonoGCHandleData &p_other) {
#ifdef DEBUG_ENABLED
		CRASH_COND(!is_released());
#endif
		handle = p_other.handle;
		type = p_other.type;
		return *this;
	}

	MonoGCHandleData(const MonoGCHandleData &) = default;

	MonoGCHandleData() {}

	MonoGCHandleData(uint32_t p_handle, gdmono::GCHandleType p_type) :
			handle(p_handle),
			type(p_type) {
	}

	static MonoGCHandleData new_strong_handle(MonoObject *p_object);
	static MonoGCHandleData new_strong_handle_pinned(MonoObject *p_object);
	static MonoGCHandleData new_weak_handle(MonoObject *p_object);
};

class MonoGCHandleRef : public Reference {
	GDCLASS(MonoGCHandleRef, Reference);

	MonoGCHandleData data;

public:
	static Ref<MonoGCHandleRef> create_strong(MonoObject *p_object);
	static Ref<MonoGCHandleRef> create_weak(MonoObject *p_object);

	_FORCE_INLINE_ bool is_released() const { return data.is_released(); }
	_FORCE_INLINE_ bool is_weak() const { return data.is_weak(); }

	_FORCE_INLINE_ MonoObject *get_target() const { return data.get_target(); }

	void release() { data.release(); }

	_FORCE_INLINE_ void set_handle(uint32_t p_handle, gdmono::GCHandleType p_handle_type) {
		data = MonoGCHandleData(p_handle, p_handle_type);
	}

	MonoGCHandleRef(const MonoGCHandleData &p_gc_handle_data) :
			data(p_gc_handle_data) {
	}
	~MonoGCHandleRef() { release(); }
};

#endif // CSHARP_GC_HANDLE_H
