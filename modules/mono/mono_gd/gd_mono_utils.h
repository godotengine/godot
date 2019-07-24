/*************************************************************************/
/*  gd_mono_utils.h                                                      */
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

#ifndef GD_MONOUTILS_H
#define GD_MONOUTILS_H

#include <mono/metadata/threads.h>

#include "../mono_gc_handle.h"
#include "../utils/macros.h"
#include "../utils/thread_local.h"
#include "gd_mono_header.h"

#include "core/object.h"
#include "core/reference.h"

#define UNHANDLED_EXCEPTION(m_exc)                     \
	if (unlikely(m_exc != NULL)) {                     \
		GDMonoUtils::debug_unhandled_exception(m_exc); \
		GD_UNREACHABLE();                              \
	}

namespace GDMonoUtils {

typedef void (*GodotObject_Dispose)(MonoObject *, MonoException **);
typedef Array *(*Array_GetPtr)(MonoObject *, MonoException **);
typedef Dictionary *(*Dictionary_GetPtr)(MonoObject *, MonoException **);
typedef MonoObject *(*SignalAwaiter_SignalCallback)(MonoObject *, MonoArray *, MonoException **);
typedef MonoObject *(*SignalAwaiter_FailureCallback)(MonoObject *, MonoException **);
typedef MonoObject *(*GodotTaskScheduler_Activate)(MonoObject *, MonoException **);
typedef MonoArray *(*StackTrace_GetFrames)(MonoObject *, MonoException **);
typedef void (*DebugUtils_StackFrameInfo)(MonoObject *, MonoString **, int *, MonoString **, MonoException **);

typedef MonoBoolean (*TypeIsGenericArray)(MonoReflectionType *, MonoException **);
typedef MonoBoolean (*TypeIsGenericDictionary)(MonoReflectionType *, MonoException **);

typedef void (*ArrayGetElementType)(MonoReflectionType *, MonoReflectionType **, MonoException **);
typedef void (*DictionaryGetKeyValueTypes)(MonoReflectionType *, MonoReflectionType **, MonoReflectionType **, MonoException **);

typedef MonoBoolean (*GenericIEnumerableIsAssignableFromType)(MonoReflectionType *, MonoException **);
typedef MonoBoolean (*GenericIDictionaryIsAssignableFromType)(MonoReflectionType *, MonoException **);
typedef MonoBoolean (*GenericIEnumerableIsAssignableFromType_with_info)(MonoReflectionType *, MonoReflectionType **, MonoException **);
typedef MonoBoolean (*GenericIDictionaryIsAssignableFromType_with_info)(MonoReflectionType *, MonoReflectionType **, MonoReflectionType **, MonoException **);

typedef MonoReflectionType *(*MakeGenericArrayType)(MonoReflectionType *, MonoException **);
typedef MonoReflectionType *(*MakeGenericDictionaryType)(MonoReflectionType *, MonoReflectionType *, MonoException **);

typedef void (*EnumerableToArray)(MonoObject *, Array *, MonoException **);
typedef void (*IDictionaryToDictionary)(MonoObject *, Dictionary *, MonoException **);
typedef void (*GenericIDictionaryToDictionary)(MonoObject *, Dictionary *, MonoException **);

namespace Marshal {

bool type_is_generic_array(MonoReflectionType *p_reftype);
bool type_is_generic_dictionary(MonoReflectionType *p_reftype);

void array_get_element_type(MonoReflectionType *p_array_reftype, MonoReflectionType **r_elem_reftype);
void dictionary_get_key_value_types(MonoReflectionType *p_dict_reftype, MonoReflectionType **r_key_reftype, MonoReflectionType **r_value_reftype);

bool generic_ienumerable_is_assignable_from(MonoReflectionType *p_reftype);
bool generic_idictionary_is_assignable_from(MonoReflectionType *p_reftype);
bool generic_ienumerable_is_assignable_from(MonoReflectionType *p_reftype, MonoReflectionType **r_elem_reftype);
bool generic_idictionary_is_assignable_from(MonoReflectionType *p_reftype, MonoReflectionType **r_key_reftype, MonoReflectionType **r_value_reftype);

GDMonoClass *make_generic_array_type(MonoReflectionType *p_elem_reftype);
GDMonoClass *make_generic_dictionary_type(MonoReflectionType *p_key_reftype, MonoReflectionType *p_value_reftype);

Array enumerable_to_array(MonoObject *p_enumerable);
Dictionary idictionary_to_dictionary(MonoObject *p_idictionary);
Dictionary generic_idictionary_to_dictionary(MonoObject *p_generic_idictionary);

} // namespace Marshal

// End of MarshalUtils methods

struct MonoCache {

	// -----------------------------------------------
	// corlib classes

	// Let's use the no-namespace format for these too
	GDMonoClass *class_MonoObject;
	GDMonoClass *class_bool;
	GDMonoClass *class_int8_t;
	GDMonoClass *class_int16_t;
	GDMonoClass *class_int32_t;
	GDMonoClass *class_int64_t;
	GDMonoClass *class_uint8_t;
	GDMonoClass *class_uint16_t;
	GDMonoClass *class_uint32_t;
	GDMonoClass *class_uint64_t;
	GDMonoClass *class_float;
	GDMonoClass *class_double;
	GDMonoClass *class_String;
	GDMonoClass *class_IntPtr;

	GDMonoClass *class_System_Collections_IEnumerable;
	GDMonoClass *class_System_Collections_IDictionary;

#ifdef DEBUG_ENABLED
	GDMonoClass *class_System_Diagnostics_StackTrace;
	StackTrace_GetFrames methodthunk_System_Diagnostics_StackTrace_GetFrames;
	GDMonoMethod *method_System_Diagnostics_StackTrace_ctor_bool;
	GDMonoMethod *method_System_Diagnostics_StackTrace_ctor_Exception_bool;
#endif

	GDMonoClass *class_KeyNotFoundException;

	MonoClass *rawclass_Dictionary;
	// -----------------------------------------------

	GDMonoClass *class_Vector2;
	GDMonoClass *class_Rect2;
	GDMonoClass *class_Transform2D;
	GDMonoClass *class_Vector3;
	GDMonoClass *class_Basis;
	GDMonoClass *class_Quat;
	GDMonoClass *class_Transform;
	GDMonoClass *class_AABB;
	GDMonoClass *class_Color;
	GDMonoClass *class_Plane;
	GDMonoClass *class_NodePath;
	GDMonoClass *class_RID;
	GDMonoClass *class_GodotObject;
	GDMonoClass *class_GodotResource;
	GDMonoClass *class_Node;
	GDMonoClass *class_Control;
	GDMonoClass *class_Spatial;
	GDMonoClass *class_WeakRef;
	GDMonoClass *class_Array;
	GDMonoClass *class_Dictionary;
	GDMonoClass *class_MarshalUtils;
	GDMonoClass *class_ISerializationListener;

#ifdef DEBUG_ENABLED
	GDMonoClass *class_DebuggingUtils;
	DebugUtils_StackFrameInfo methodthunk_DebuggingUtils_GetStackFrameInfo;
#endif

	GDMonoClass *class_ExportAttribute;
	GDMonoField *field_ExportAttribute_hint;
	GDMonoField *field_ExportAttribute_hintString;
	GDMonoClass *class_SignalAttribute;
	GDMonoClass *class_ToolAttribute;
	GDMonoClass *class_RemoteAttribute;
	GDMonoClass *class_SyncAttribute;
	GDMonoClass *class_RemoteSyncAttribute;
	GDMonoClass *class_MasterSyncAttribute;
	GDMonoClass *class_PuppetSyncAttribute;
	GDMonoClass *class_MasterAttribute;
	GDMonoClass *class_PuppetAttribute;
	GDMonoClass *class_SlaveAttribute;
	GDMonoClass *class_GodotMethodAttribute;
	GDMonoField *field_GodotMethodAttribute_methodName;

	GDMonoField *field_GodotObject_ptr;
	GDMonoField *field_NodePath_ptr;
	GDMonoField *field_Image_ptr;
	GDMonoField *field_RID_ptr;

	GodotObject_Dispose methodthunk_GodotObject_Dispose;
	Array_GetPtr methodthunk_Array_GetPtr;
	Dictionary_GetPtr methodthunk_Dictionary_GetPtr;
	SignalAwaiter_SignalCallback methodthunk_SignalAwaiter_SignalCallback;
	SignalAwaiter_FailureCallback methodthunk_SignalAwaiter_FailureCallback;
	GodotTaskScheduler_Activate methodthunk_GodotTaskScheduler_Activate;

	// Start of MarshalUtils methods

	TypeIsGenericArray methodthunk_MarshalUtils_TypeIsGenericArray;
	TypeIsGenericDictionary methodthunk_MarshalUtils_TypeIsGenericDictionary;

	ArrayGetElementType methodthunk_MarshalUtils_ArrayGetElementType;
	DictionaryGetKeyValueTypes methodthunk_MarshalUtils_DictionaryGetKeyValueTypes;

	GenericIEnumerableIsAssignableFromType methodthunk_MarshalUtils_GenericIEnumerableIsAssignableFromType;
	GenericIDictionaryIsAssignableFromType methodthunk_MarshalUtils_GenericIDictionaryIsAssignableFromType;
	GenericIEnumerableIsAssignableFromType_with_info methodthunk_MarshalUtils_GenericIEnumerableIsAssignableFromType_with_info;
	GenericIDictionaryIsAssignableFromType_with_info methodthunk_MarshalUtils_GenericIDictionaryIsAssignableFromType_with_info;

	MakeGenericArrayType methodthunk_MarshalUtils_MakeGenericArrayType;
	MakeGenericDictionaryType methodthunk_MarshalUtils_MakeGenericDictionaryType;

	EnumerableToArray methodthunk_MarshalUtils_EnumerableToArray;
	IDictionaryToDictionary methodthunk_MarshalUtils_IDictionaryToDictionary;
	GenericIDictionaryToDictionary methodthunk_MarshalUtils_GenericIDictionaryToDictionary;

	// End of MarshalUtils methods

	Ref<MonoGCHandle> task_scheduler_handle;

	bool corlib_cache_updated;
	bool godot_api_cache_updated;

	void clear_corlib_cache();
	void clear_godot_api_cache();

	MonoCache() {
		clear_corlib_cache();
		clear_godot_api_cache();
	}
};

extern MonoCache mono_cache;

void update_corlib_cache();
void update_godot_api_cache();

inline void clear_corlib_cache() {
	mono_cache.clear_corlib_cache();
}

inline void clear_godot_api_cache() {
	mono_cache.clear_godot_api_cache();
}

_FORCE_INLINE_ bool tools_godot_api_check() {
#ifdef TOOLS_ENABLED
	return mono_cache.godot_api_cache_updated;
#else
	return true; // Assume it's updated if this was called, otherwise it's a bug
#endif
}

_FORCE_INLINE_ void hash_combine(uint32_t &p_hash, const uint32_t &p_with_hash) {
	p_hash ^= p_with_hash + 0x9e3779b9 + (p_hash << 6) + (p_hash >> 2);
}

/**
 * If the object has a csharp script, returns the target of the gchandle stored in the script instance
 * Otherwise returns a newly constructed MonoObject* which is attached to the object
 * Returns NULL on error
 */
MonoObject *unmanaged_get_managed(Object *unmanaged);

void set_main_thread(MonoThread *p_thread);
void attach_current_thread();
void detach_current_thread();
MonoThread *get_current_thread();

_FORCE_INLINE_ bool is_main_thread() {
	return mono_domain_get() != NULL && mono_thread_get_main() == mono_thread_current();
}

void runtime_object_init(MonoObject *p_this_obj, GDMonoClass *p_class, MonoException **r_exc = NULL);

GDMonoClass *get_object_class(MonoObject *p_object);
GDMonoClass *type_get_proxy_class(const StringName &p_type);
GDMonoClass *get_class_native_base(GDMonoClass *p_class);

MonoObject *create_managed_for_godot_object(GDMonoClass *p_class, const StringName &p_native, Object *p_object);

MonoObject *create_managed_from(const NodePath &p_from);
MonoObject *create_managed_from(const RID &p_from);
MonoObject *create_managed_from(const Array &p_from, GDMonoClass *p_class);
MonoObject *create_managed_from(const Dictionary &p_from, GDMonoClass *p_class);

MonoDomain *create_domain(const String &p_friendly_name);

String get_exception_name_and_message(MonoException *p_exc);
void set_exception_message(MonoException *p_exc, String message);

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

extern _THREAD_LOCAL_(int) current_invoke_count;

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

} // namespace GDMonoUtils

#define NATIVE_GDMONOCLASS_NAME(m_class) (GDMonoMarshal::mono_string_to_godot((MonoString *)m_class->get_field(BINDINGS_NATIVE_NAME_FIELD)->get_value(NULL)))

#define CACHED_CLASS(m_class) (GDMonoUtils::mono_cache.class_##m_class)
#define CACHED_CLASS_RAW(m_class) (GDMonoUtils::mono_cache.class_##m_class->get_mono_ptr())
#define CACHED_RAW_MONO_CLASS(m_class) (GDMonoUtils::mono_cache.rawclass_##m_class)
#define CACHED_FIELD(m_class, m_field) (GDMonoUtils::mono_cache.field_##m_class##_##m_field)
#define CACHED_METHOD(m_class, m_method) (GDMonoUtils::mono_cache.method_##m_class##_##m_method)
#define CACHED_METHOD_THUNK(m_class, m_method) (GDMonoUtils::mono_cache.methodthunk_##m_class##_##m_method)
#define CACHED_PROPERTY(m_class, m_property) (GDMonoUtils::mono_cache.property_##m_class##_##m_property)

#ifdef REAL_T_IS_DOUBLE
#define REAL_T_MONOCLASS CACHED_CLASS_RAW(double)
#else
#define REAL_T_MONOCLASS CACHED_CLASS_RAW(float)
#endif

#define GD_MONO_BEGIN_RUNTIME_INVOKE                                              \
	int &_runtime_invoke_count_ref = GDMonoUtils::get_runtime_invoke_count_ref(); \
	_runtime_invoke_count_ref += 1;

#define GD_MONO_END_RUNTIME_INVOKE \
	_runtime_invoke_count_ref -= 1;

inline void invoke_method_thunk(void (*p_method_thunk)()) {
	GD_MONO_BEGIN_RUNTIME_INVOKE;
	p_method_thunk();
	GD_MONO_END_RUNTIME_INVOKE;
}

template <class R>
R invoke_method_thunk(R (*p_method_thunk)()) {
	GD_MONO_BEGIN_RUNTIME_INVOKE;
	R r = p_method_thunk();
	GD_MONO_END_RUNTIME_INVOKE;
	return r;
}

template <class P1>
void invoke_method_thunk(void (*p_method_thunk)(P1), P1 p_arg1) {
	GD_MONO_BEGIN_RUNTIME_INVOKE;
	p_method_thunk(p_arg1);
	GD_MONO_END_RUNTIME_INVOKE;
}

template <class R, class P1>
R invoke_method_thunk(R (*p_method_thunk)(P1), P1 p_arg1) {
	GD_MONO_BEGIN_RUNTIME_INVOKE;
	R r = p_method_thunk(p_arg1);
	GD_MONO_END_RUNTIME_INVOKE;
	return r;
}

template <class P1, class P2>
void invoke_method_thunk(void (*p_method_thunk)(P1, P2), P1 p_arg1, P2 p_arg2) {
	GD_MONO_BEGIN_RUNTIME_INVOKE;
	p_method_thunk(p_arg1, p_arg2);
	GD_MONO_END_RUNTIME_INVOKE;
}

template <class R, class P1, class P2>
R invoke_method_thunk(R (*p_method_thunk)(P1, P2), P1 p_arg1, P2 p_arg2) {
	GD_MONO_BEGIN_RUNTIME_INVOKE;
	R r = p_method_thunk(p_arg1, p_arg2);
	GD_MONO_END_RUNTIME_INVOKE;
	return r;
}

template <class P1, class P2, class P3>
void invoke_method_thunk(void (*p_method_thunk)(P1, P2, P3), P1 p_arg1, P2 p_arg2, P3 p_arg3) {
	GD_MONO_BEGIN_RUNTIME_INVOKE;
	p_method_thunk(p_arg1, p_arg2, p_arg3);
	GD_MONO_END_RUNTIME_INVOKE;
}

template <class R, class P1, class P2, class P3>
R invoke_method_thunk(R (*p_method_thunk)(P1, P2, P3), P1 p_arg1, P2 p_arg2, P3 p_arg3) {
	GD_MONO_BEGIN_RUNTIME_INVOKE;
	R r = p_method_thunk(p_arg1, p_arg2, p_arg3);
	GD_MONO_END_RUNTIME_INVOKE;
	return r;
}

template <class P1, class P2, class P3, class P4>
void invoke_method_thunk(void (*p_method_thunk)(P1, P2, P3, P4), P1 p_arg1, P2 p_arg2, P3 p_arg3, P4 p_arg4) {
	GD_MONO_BEGIN_RUNTIME_INVOKE;
	p_method_thunk(p_arg1, p_arg2, p_arg3, p_arg4);
	GD_MONO_END_RUNTIME_INVOKE;
}

template <class R, class P1, class P2, class P3, class P4>
R invoke_method_thunk(R (*p_method_thunk)(P1, P2, P3, P4), P1 p_arg1, P2 p_arg2, P3 p_arg3, P4 p_arg4) {
	GD_MONO_BEGIN_RUNTIME_INVOKE;
	R r = p_method_thunk(p_arg1, p_arg2, p_arg3, p_arg4);
	GD_MONO_END_RUNTIME_INVOKE;
	return r;
}

template <class P1, class P2, class P3, class P4, class P5>
void invoke_method_thunk(void (*p_method_thunk)(P1, P2, P3, P4, P5), P1 p_arg1, P2 p_arg2, P3 p_arg3, P4 p_arg4, P5 p_arg5) {
	GD_MONO_BEGIN_RUNTIME_INVOKE;
	p_method_thunk(p_arg1, p_arg2, p_arg3, p_arg4, p_arg5);
	GD_MONO_END_RUNTIME_INVOKE;
}

template <class R, class P1, class P2, class P3, class P4, class P5>
R invoke_method_thunk(R (*p_method_thunk)(P1, P2, P3, P4, P5), P1 p_arg1, P2 p_arg2, P3 p_arg3, P4 p_arg4, P5 p_arg5) {
	GD_MONO_BEGIN_RUNTIME_INVOKE;
	R r = p_method_thunk(p_arg1, p_arg2, p_arg3, p_arg4, p_arg5);
	GD_MONO_END_RUNTIME_INVOKE;
	return r;
}

#endif // GD_MONOUTILS_H
