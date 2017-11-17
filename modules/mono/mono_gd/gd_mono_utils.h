/*************************************************************************/
/*  gd_mono_utils.h                                                      */
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
#ifndef GD_MONOUTILS_H
#define GD_MONOUTILS_H

#include <mono/metadata/threads.h>

#include "../mono_gc_handle.h"
#include "gd_mono_header.h"

#include "object.h"
#include "reference.h"

namespace GDMonoUtils {

typedef MonoObject *(*MarshalUtils_DictToArrays)(MonoObject *, MonoArray **, MonoArray **, MonoObject **);
typedef MonoObject *(*MarshalUtils_ArraysToDict)(MonoArray *, MonoArray *, MonoObject **);
typedef MonoObject *(*SignalAwaiter_SignalCallback)(MonoObject *, MonoArray **, MonoObject **);
typedef MonoObject *(*SignalAwaiter_FailureCallback)(MonoObject *, MonoObject **);
typedef MonoObject *(*GodotTaskScheduler_Activate)(MonoObject *, MonoObject **);

struct MonoCache {
	// Format for cached classes in the Godot namespace: class_<Class>
	// Macro: CACHED_CLASS(<Class>)

	// Format for cached classes in a different namespace: class_<Namespace>_<Class>
	// Macro: CACHED_NS_CLASS(<Namespace>, <Class>)

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
	GDMonoClass *class_GodotReference;
	GDMonoClass *class_Node;
	GDMonoClass *class_Control;
	GDMonoClass *class_Spatial;
	GDMonoClass *class_WeakRef;
	GDMonoClass *class_MarshalUtils;

	GDMonoClass *class_ExportAttribute;
	GDMonoField *field_ExportAttribute_hint;
	GDMonoField *field_ExportAttribute_hint_string;
	GDMonoClass *class_ToolAttribute;
	GDMonoClass *class_RemoteAttribute;
	GDMonoClass *class_SyncAttribute;
	GDMonoClass *class_MasterAttribute;
	GDMonoClass *class_SlaveAttribute;
	GDMonoClass *class_GodotMethodAttribute;
	GDMonoField *field_GodotMethodAttribute_methodName;

	GDMonoField *field_GodotObject_ptr;
	GDMonoField *field_NodePath_ptr;
	GDMonoField *field_Image_ptr;
	GDMonoField *field_RID_ptr;

	MarshalUtils_DictToArrays methodthunk_MarshalUtils_DictionaryToArrays;
	MarshalUtils_ArraysToDict methodthunk_MarshalUtils_ArraysToDictionary;
	SignalAwaiter_SignalCallback methodthunk_SignalAwaiter_SignalCallback;
	SignalAwaiter_FailureCallback methodthunk_SignalAwaiter_FailureCallback;
	GodotTaskScheduler_Activate methodthunk_GodotTaskScheduler_Activate;

	Ref<MonoGCHandle> task_scheduler_handle;

	void clear_members();
	void cleanup() {}

	MonoCache() {
		clear_members();
	}
};

extern MonoCache mono_cache;

void update_corlib_cache();
void update_godot_api_cache();
void clear_cache();

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

GDMonoClass *get_object_class(MonoObject *p_object);
GDMonoClass *type_get_proxy_class(const StringName &p_type);
GDMonoClass *get_class_native_base(GDMonoClass *p_class);

MonoObject *create_managed_for_godot_object(GDMonoClass *p_class, const StringName &p_native, Object *p_object);

MonoObject *create_managed_from(const NodePath &p_from);
MonoObject *create_managed_from(const RID &p_from);

MonoDomain *create_domain(const String &p_friendly_name);

String get_exception_name_and_message(MonoObject *p_ex);

} // namespace GDMonoUtils

#define NATIVE_GDMONOCLASS_NAME(m_class) (GDMonoMarshal::mono_string_to_godot((MonoString *)m_class->get_field(BINDINGS_NATIVE_NAME_FIELD)->get_value(NULL)))

#define CACHED_CLASS(m_class) (GDMonoUtils::mono_cache.class_##m_class)
#define CACHED_CLASS_RAW(m_class) (GDMonoUtils::mono_cache.class_##m_class->get_raw())
#define CACHED_NS_CLASS(m_ns, m_class) (GDMonoUtils::mono_cache.class_##m_ns##_##m_class)
#define CACHED_RAW_MONO_CLASS(m_class) (GDMonoUtils::mono_cache.rawclass_##m_class)
#define CACHED_FIELD(m_class, m_field) (GDMonoUtils::mono_cache.field_##m_class##_##m_field)
#define CACHED_METHOD_THUNK(m_class, m_method) (GDMonoUtils::mono_cache.methodthunk_##m_class##_##m_method)

#ifdef REAL_T_IS_DOUBLE
#define REAL_T_MONOCLASS CACHED_CLASS_RAW(double)
#else
#define REAL_T_MONOCLASS CACHED_CLASS_RAW(float)
#endif

#endif // GD_MONOUTILS_H
