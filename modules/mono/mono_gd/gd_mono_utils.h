/*************************************************************************/
/*  gd_mono_utils.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GD_MONOUTILS_H
#define GD_MONOUTILS_H

#include <mono/metadata/threads.h>

#include "../mono_gc_handle.h"
#include "../utils/macros.h"
#include "gd_mono_header.h"

#include "core/class_db.h"
#include "core/reference.h"

#define UNHANDLED_EXCEPTION(m_exc)                     \
	if (unlikely(m_exc != nullptr)) {                  \
		GDMonoUtils::debug_unhandled_exception(m_exc); \
		GD_UNREACHABLE();                              \
	} else                                             \
		((void)0)

namespace GDMonoUtils {

namespace Marshal {

bool type_is_generic_array(MonoReflectionType *p_reftype);
bool type_is_generic_dictionary(MonoReflectionType *p_reftype);
bool type_is_system_generic_list(MonoReflectionType *p_reftype);
bool type_is_system_generic_dictionary(MonoReflectionType *p_reftype);
bool type_is_generic_ienumerable(MonoReflectionType *p_reftype);
bool type_is_generic_icollection(MonoReflectionType *p_reftype);
bool type_is_generic_idictionary(MonoReflectionType *p_reftype);

void array_get_element_type(MonoReflectionType *p_array_reftype, MonoReflectionType **r_elem_reftype);
void dictionary_get_key_value_types(MonoReflectionType *p_dict_reftype, MonoReflectionType **r_key_reftype, MonoReflectionType **r_value_reftype);

GDMonoClass *make_generic_array_type(MonoReflectionType *p_elem_reftype);
GDMonoClass *make_generic_dictionary_type(MonoReflectionType *p_key_reftype, MonoReflectionType *p_value_reftype);

} // namespace Marshal

_FORCE_INLINE_ void hash_combine(uint32_t &p_hash, const uint32_t &p_with_hash) {
	p_hash ^= p_with_hash + 0x9e3779b9 + (p_hash << 6) + (p_hash >> 2);
}

/**
 * If the object has a csharp script, returns the target of the gchandle stored in the script instance
 * Otherwise returns a newly constructed MonoObject* which is attached to the object
 * Returns nullptr on error
 */
MonoObject *unmanaged_get_managed(Object *unmanaged);

void set_main_thread(MonoThread *p_thread);
MonoThread *attach_current_thread();
void detach_current_thread();
void detach_current_thread(MonoThread *p_mono_thread);
MonoThread *get_current_thread();
bool is_thread_attached();

uint32_t new_strong_gchandle(MonoObject *p_object);
uint32_t new_strong_gchandle_pinned(MonoObject *p_object);
uint32_t new_weak_gchandle(MonoObject *p_object);
void free_gchandle(uint32_t p_gchandle);

void runtime_object_init(MonoObject *p_this_obj, GDMonoClass *p_class, MonoException **r_exc = nullptr);

bool mono_delegate_equal(MonoDelegate *p_a, MonoDelegate *p_b);

GDMonoClass *get_object_class(MonoObject *p_object);
GDMonoClass *type_get_proxy_class(const StringName &p_type);
GDMonoClass *get_class_native_base(GDMonoClass *p_class);

MonoObject *create_managed_for_godot_object(GDMonoClass *p_class, const StringName &p_native, Object *p_object);

MonoObject *create_managed_from(const StringName &p_from);
MonoObject *create_managed_from(const NodePath &p_from);
MonoObject *create_managed_from(const RID &p_from);
MonoObject *create_managed_from(const Array &p_from, GDMonoClass *p_class);
MonoObject *create_managed_from(const Dictionary &p_from, GDMonoClass *p_class);

MonoDomain *create_domain(const String &p_friendly_name);

String get_type_desc(MonoType *p_type);
String get_type_desc(MonoReflectionType *p_reftype);

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

MonoObject *runtime_invoke(MonoMethod *p_method, void *p_obj, void **p_params, MonoException **r_exc);
MonoObject *runtime_invoke_array(MonoMethod *p_method, void *p_obj, MonoArray *p_params, MonoException **r_exc);

MonoString *object_to_string(MonoObject *p_obj, MonoException **r_exc);

void property_set_value(MonoProperty *p_prop, void *p_obj, void **p_params, MonoException **r_exc);
MonoObject *property_get_value(MonoProperty *p_prop, void *p_obj, void **p_params, MonoException **r_exc);

uint64_t unbox_enum_value(MonoObject *p_boxed, MonoType *p_enum_basetype, bool &r_error);

void dispose(MonoObject *p_mono_object, MonoException **r_exc);

struct ScopeThreadAttach {
	ScopeThreadAttach();
	~ScopeThreadAttach();

private:
	MonoThread *mono_thread = nullptr;
};

StringName get_native_godot_class_name(GDMonoClass *p_class);

} // namespace GDMonoUtils

#define NATIVE_GDMONOCLASS_NAME(m_class) (GDMonoUtils::get_native_godot_class_name(m_class))

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

#endif // GD_MONOUTILS_H
