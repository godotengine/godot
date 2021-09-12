/*************************************************************************/
/*  gd_mono_utils.h                                                      */
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

#ifndef GD_MONO_UTILS_H
#define GD_MONO_UTILS_H

#include <mono/metadata/threads.h>

#include "../mono_gc_handle.h"
#include "../utils/macros.h"
#ifdef JAVASCRIPT_ENABLED
#include "gd_mono_wasm_m2n.h"
#endif

#include "core/object/class_db.h"
#include "core/object/ref_counted.h"

#define UNHANDLED_EXCEPTION(m_exc)                     \
	if (unlikely(m_exc != nullptr)) {                  \
		GDMonoUtils::debug_unhandled_exception(m_exc); \
		GD_UNREACHABLE();                              \
	} else                                             \
		((void)0)

namespace GDMonoUtils {

namespace Marshal {
bool type_has_flags_attribute(MonoReflectionType *p_reftype);
} // namespace Marshal

_FORCE_INLINE_ void hash_combine(uint32_t &p_hash, const uint32_t &p_with_hash) {
	p_hash ^= p_with_hash + 0x9e3779b9 + (p_hash << 6) + (p_hash >> 2);
}

void set_main_thread(MonoThread *p_thread);
MonoThread *attach_current_thread();
void detach_current_thread();
void detach_current_thread(MonoThread *p_mono_thread);
MonoThread *get_current_thread();
bool is_thread_attached();

MonoDomain *create_domain(const String &p_friendly_name);

String get_exception_name_and_message(MonoException *p_exc);

void debug_print_unhandled_exception(MonoException *p_exc);
void debug_send_unhandled_exception_error(MonoException *p_exc);
void debug_unhandled_exception(MonoException *p_exc);
void print_unhandled_exception(MonoException *p_exc);

/**
 * Sets the exception as pending. The exception will be thrown when returning to managed code.
 * If no managed method is being invoked by the runtime, the exception will be treated as
 * an unhandled exception and the method will not return.
 */
void set_pending_exception(MonoException *p_exc);

extern thread_local int current_invoke_count;

_FORCE_INLINE_ int get_runtime_invoke_count() {
	return current_invoke_count;
}

_FORCE_INLINE_ int &get_runtime_invoke_count_ref() {
	return current_invoke_count;
}

uint64_t unbox_enum_value(MonoObject *p_boxed, MonoType *p_enum_basetype, bool &r_error);

struct ScopeThreadAttach {
	ScopeThreadAttach();
	~ScopeThreadAttach();

private:
	MonoThread *mono_thread = nullptr;
};

template <typename... P>
void add_internal_call(const char *p_name, void (*p_func)(P...)) {
#ifdef JAVASCRIPT_ENABLED
	GDMonoWasmM2n::ICallTrampolines<P...>::add();
#endif
	mono_add_internal_call(p_name, (void *)p_func);
}

template <typename R, typename... P>
void add_internal_call(const char *p_name, R (*p_func)(P...)) {
#ifdef JAVASCRIPT_ENABLED
	GDMonoWasmM2n::ICallTrampolinesR<R, P...>::add();
#endif
	mono_add_internal_call(p_name, (void *)p_func);
}
} // namespace GDMonoUtils

#define GD_MONO_BEGIN_RUNTIME_INVOKE                                              \
	int &_runtime_invoke_count_ref = GDMonoUtils::get_runtime_invoke_count_ref(); \
	_runtime_invoke_count_ref += 1;                                               \
	((void)0)

#define GD_MONO_END_RUNTIME_INVOKE  \
	_runtime_invoke_count_ref -= 1; \
	((void)0)

#define GD_MONO_SCOPE_THREAD_ATTACH                                   \
	GDMonoUtils::ScopeThreadAttach __gdmono__scope__thread__attach__; \
	(void)__gdmono__scope__thread__attach__;                          \
	((void)0)

#ifdef DEBUG_ENABLED
#define GD_MONO_ASSERT_THREAD_ATTACHED              \
	CRASH_COND(!GDMonoUtils::is_thread_attached()); \
	((void)0)
#else
#define GD_MONO_ASSERT_THREAD_ATTACHED ((void)0)
#endif

#endif // GD_MONO_UTILS_H
