/*************************************************************************/
/*  gd_mono_cache.cpp                                                    */
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

#include "gd_mono_cache.h"

#include "gd_mono.h"
#include "gd_mono_class.h"
#include "gd_mono_marshal.h"
#include "gd_mono_method.h"
#include "gd_mono_utils.h"

namespace GDMonoCache {

CachedData cached_data;

#define CACHE_AND_CHECK(m_var, m_val)                                               \
	{                                                                               \
		CRASH_COND(m_var != NULL);                                                  \
		m_var = m_val;                                                              \
		ERR_FAIL_COND_MSG(m_var == NULL, "Mono Cache: Member " #m_var " is null."); \
	}

#define CACHE_CLASS_AND_CHECK(m_class, m_val) CACHE_AND_CHECK(cached_data.class_##m_class, m_val)
#define CACHE_NS_CLASS_AND_CHECK(m_ns, m_class, m_val) CACHE_AND_CHECK(cached_data.class_##m_ns##_##m_class, m_val)
#define CACHE_RAW_MONO_CLASS_AND_CHECK(m_class, m_val) CACHE_AND_CHECK(cached_data.rawclass_##m_class, m_val)
#define CACHE_FIELD_AND_CHECK(m_class, m_field, m_val) CACHE_AND_CHECK(cached_data.field_##m_class##_##m_field, m_val)
#define CACHE_METHOD_AND_CHECK(m_class, m_method, m_val) CACHE_AND_CHECK(cached_data.method_##m_class##_##m_method, m_val)
#define CACHE_PROPERTY_AND_CHECK(m_class, m_property, m_val) CACHE_AND_CHECK(cached_data.property_##m_class##_##m_property, m_val)

#define CACHE_METHOD_THUNK_AND_CHECK_IMPL(m_var, m_val)                                        \
	{                                                                                          \
		CRASH_COND(!m_var.is_null());                                                          \
		ERR_FAIL_COND_MSG(m_val == NULL, "Mono Cache: Method for member " #m_var " is null."); \
		m_var.set_from_method(m_val);                                                          \
		ERR_FAIL_COND_MSG(m_var.is_null(), "Mono Cache: Member " #m_var " is null.");          \
	}

#define CACHE_METHOD_THUNK_AND_CHECK(m_class, m_method, m_val) CACHE_METHOD_THUNK_AND_CHECK_IMPL(cached_data.methodthunk_##m_class##_##m_method, m_val)

void CachedData::clear_corlib_cache() {
	corlib_cache_updated = false;

	class_MonoObject = NULL;
	class_bool = NULL;
	class_int8_t = NULL;
	class_int16_t = NULL;
	class_int32_t = NULL;
	class_int64_t = NULL;
	class_uint8_t = NULL;
	class_uint16_t = NULL;
	class_uint32_t = NULL;
	class_uint64_t = NULL;
	class_float = NULL;
	class_double = NULL;
	class_String = NULL;
	class_IntPtr = NULL;

	class_System_Collections_IEnumerable = NULL;
	class_System_Collections_ICollection = NULL;
	class_System_Collections_IDictionary = NULL;

#ifdef DEBUG_ENABLED
	class_System_Diagnostics_StackTrace = NULL;
	methodthunk_System_Diagnostics_StackTrace_GetFrames.nullify();
	method_System_Diagnostics_StackTrace_ctor_bool = NULL;
	method_System_Diagnostics_StackTrace_ctor_Exception_bool = NULL;
#endif

	class_KeyNotFoundException = NULL;
}

void CachedData::clear_godot_api_cache() {
	godot_api_cache_updated = false;

	rawclass_Dictionary = NULL;

	class_Vector2 = NULL;
	class_Rect2 = NULL;
	class_Transform2D = NULL;
	class_Vector3 = NULL;
	class_Basis = NULL;
	class_Quat = NULL;
	class_Transform = NULL;
	class_AABB = NULL;
	class_Color = NULL;
	class_Plane = NULL;
	class_NodePath = NULL;
	class_RID = NULL;
	class_GodotObject = NULL;
	class_GodotResource = NULL;
	class_Node = NULL;
	class_Control = NULL;
	class_Spatial = NULL;
	class_WeakRef = NULL;
	class_Array = NULL;
	class_Dictionary = NULL;
	class_MarshalUtils = NULL;
	class_ISerializationListener = NULL;

#ifdef DEBUG_ENABLED
	class_DebuggingUtils = NULL;
	methodthunk_DebuggingUtils_GetStackFrameInfo.nullify();
#endif

	class_ExportAttribute = NULL;
	field_ExportAttribute_hint = NULL;
	field_ExportAttribute_hintString = NULL;
	class_SignalAttribute = NULL;
	class_ToolAttribute = NULL;
	class_RemoteAttribute = NULL;
	class_SyncAttribute = NULL;
	class_MasterAttribute = NULL;
	class_PuppetAttribute = NULL;
	class_SlaveAttribute = NULL;
	class_RemoteSyncAttribute = NULL;
	class_MasterSyncAttribute = NULL;
	class_PuppetSyncAttribute = NULL;
	class_GodotMethodAttribute = NULL;
	field_GodotMethodAttribute_methodName = NULL;

	field_GodotObject_ptr = NULL;
	field_NodePath_ptr = NULL;
	field_Image_ptr = NULL;
	field_RID_ptr = NULL;

	methodthunk_GodotObject_Dispose.nullify();
	methodthunk_Array_GetPtr.nullify();
	methodthunk_Dictionary_GetPtr.nullify();
	methodthunk_SignalAwaiter_SignalCallback.nullify();
	methodthunk_SignalAwaiter_FailureCallback.nullify();
	methodthunk_GodotTaskScheduler_Activate.nullify();

	// Start of MarshalUtils methods

	methodthunk_MarshalUtils_TypeIsGenericArray.nullify();
	methodthunk_MarshalUtils_TypeIsGenericDictionary.nullify();
	methodthunk_MarshalUtils_TypeIsSystemGenericList.nullify();
	methodthunk_MarshalUtils_TypeIsSystemGenericDictionary.nullify();
	methodthunk_MarshalUtils_TypeIsGenericIEnumerable.nullify();
	methodthunk_MarshalUtils_TypeIsGenericICollection.nullify();
	methodthunk_MarshalUtils_TypeIsGenericIDictionary.nullify();

	methodthunk_MarshalUtils_GetGenericTypeDefinition.nullify();

	methodthunk_MarshalUtils_ArrayGetElementType.nullify();
	methodthunk_MarshalUtils_DictionaryGetKeyValueTypes.nullify();

	methodthunk_MarshalUtils_MakeGenericArrayType.nullify();
	methodthunk_MarshalUtils_MakeGenericDictionaryType.nullify();

	// End of MarshalUtils methods

	task_scheduler_handle = Ref<MonoGCHandle>();
}

#define GODOT_API_CLASS(m_class) (GDMono::get_singleton()->get_core_api_assembly()->get_class(BINDINGS_NAMESPACE, #m_class))
#define GODOT_API_NS_CLASS(m_ns, m_class) (GDMono::get_singleton()->get_core_api_assembly()->get_class(m_ns, #m_class))

void update_corlib_cache() {
	CACHE_CLASS_AND_CHECK(MonoObject, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_object_class()));
	CACHE_CLASS_AND_CHECK(bool, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_boolean_class()));
	CACHE_CLASS_AND_CHECK(int8_t, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_sbyte_class()));
	CACHE_CLASS_AND_CHECK(int16_t, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_int16_class()));
	CACHE_CLASS_AND_CHECK(int32_t, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_int32_class()));
	CACHE_CLASS_AND_CHECK(int64_t, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_int64_class()));
	CACHE_CLASS_AND_CHECK(uint8_t, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_byte_class()));
	CACHE_CLASS_AND_CHECK(uint16_t, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_uint16_class()));
	CACHE_CLASS_AND_CHECK(uint32_t, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_uint32_class()));
	CACHE_CLASS_AND_CHECK(uint64_t, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_uint64_class()));
	CACHE_CLASS_AND_CHECK(float, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_single_class()));
	CACHE_CLASS_AND_CHECK(double, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_double_class()));
	CACHE_CLASS_AND_CHECK(String, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_string_class()));
	CACHE_CLASS_AND_CHECK(IntPtr, GDMono::get_singleton()->get_corlib_assembly()->get_class(mono_get_intptr_class()));

	CACHE_CLASS_AND_CHECK(System_Collections_IEnumerable, GDMono::get_singleton()->get_corlib_assembly()->get_class("System.Collections", "IEnumerable"));
	CACHE_CLASS_AND_CHECK(System_Collections_ICollection, GDMono::get_singleton()->get_corlib_assembly()->get_class("System.Collections", "ICollection"));
	CACHE_CLASS_AND_CHECK(System_Collections_IDictionary, GDMono::get_singleton()->get_corlib_assembly()->get_class("System.Collections", "IDictionary"));

#ifdef DEBUG_ENABLED
	CACHE_CLASS_AND_CHECK(System_Diagnostics_StackTrace, GDMono::get_singleton()->get_corlib_assembly()->get_class("System.Diagnostics", "StackTrace"));
	CACHE_METHOD_THUNK_AND_CHECK(System_Diagnostics_StackTrace, GetFrames, CACHED_CLASS(System_Diagnostics_StackTrace)->get_method("GetFrames"));
	CACHE_METHOD_AND_CHECK(System_Diagnostics_StackTrace, ctor_bool, CACHED_CLASS(System_Diagnostics_StackTrace)->get_method_with_desc("System.Diagnostics.StackTrace:.ctor(bool)", true));
	CACHE_METHOD_AND_CHECK(System_Diagnostics_StackTrace, ctor_Exception_bool, CACHED_CLASS(System_Diagnostics_StackTrace)->get_method_with_desc("System.Diagnostics.StackTrace:.ctor(System.Exception,bool)", true));
#endif

	CACHE_CLASS_AND_CHECK(KeyNotFoundException, GDMono::get_singleton()->get_corlib_assembly()->get_class("System.Collections.Generic", "KeyNotFoundException"));

	cached_data.corlib_cache_updated = true;
}

void update_godot_api_cache() {
	CACHE_CLASS_AND_CHECK(Vector2, GODOT_API_CLASS(Vector2));
	CACHE_CLASS_AND_CHECK(Rect2, GODOT_API_CLASS(Rect2));
	CACHE_CLASS_AND_CHECK(Transform2D, GODOT_API_CLASS(Transform2D));
	CACHE_CLASS_AND_CHECK(Vector3, GODOT_API_CLASS(Vector3));
	CACHE_CLASS_AND_CHECK(Basis, GODOT_API_CLASS(Basis));
	CACHE_CLASS_AND_CHECK(Quat, GODOT_API_CLASS(Quat));
	CACHE_CLASS_AND_CHECK(Transform, GODOT_API_CLASS(Transform));
	CACHE_CLASS_AND_CHECK(AABB, GODOT_API_CLASS(AABB));
	CACHE_CLASS_AND_CHECK(Color, GODOT_API_CLASS(Color));
	CACHE_CLASS_AND_CHECK(Plane, GODOT_API_CLASS(Plane));
	CACHE_CLASS_AND_CHECK(NodePath, GODOT_API_CLASS(NodePath));
	CACHE_CLASS_AND_CHECK(RID, GODOT_API_CLASS(RID));
	CACHE_CLASS_AND_CHECK(GodotObject, GODOT_API_CLASS(Object));
	CACHE_CLASS_AND_CHECK(GodotResource, GODOT_API_CLASS(Resource));
	CACHE_CLASS_AND_CHECK(Node, GODOT_API_CLASS(Node));
	CACHE_CLASS_AND_CHECK(Control, GODOT_API_CLASS(Control));
	CACHE_CLASS_AND_CHECK(Spatial, GODOT_API_CLASS(Spatial));
	CACHE_CLASS_AND_CHECK(WeakRef, GODOT_API_CLASS(WeakRef));
	CACHE_CLASS_AND_CHECK(Array, GODOT_API_NS_CLASS(BINDINGS_NAMESPACE_COLLECTIONS, Array));
	CACHE_CLASS_AND_CHECK(Dictionary, GODOT_API_NS_CLASS(BINDINGS_NAMESPACE_COLLECTIONS, Dictionary));
	CACHE_CLASS_AND_CHECK(MarshalUtils, GODOT_API_CLASS(MarshalUtils));
	CACHE_CLASS_AND_CHECK(ISerializationListener, GODOT_API_CLASS(ISerializationListener));

#ifdef DEBUG_ENABLED
	CACHE_CLASS_AND_CHECK(DebuggingUtils, GODOT_API_CLASS(DebuggingUtils));
#endif

	// Attributes
	CACHE_CLASS_AND_CHECK(ExportAttribute, GODOT_API_CLASS(ExportAttribute));
	CACHE_FIELD_AND_CHECK(ExportAttribute, hint, CACHED_CLASS(ExportAttribute)->get_field("hint"));
	CACHE_FIELD_AND_CHECK(ExportAttribute, hintString, CACHED_CLASS(ExportAttribute)->get_field("hintString"));
	CACHE_CLASS_AND_CHECK(SignalAttribute, GODOT_API_CLASS(SignalAttribute));
	CACHE_CLASS_AND_CHECK(ToolAttribute, GODOT_API_CLASS(ToolAttribute));
	CACHE_CLASS_AND_CHECK(RemoteAttribute, GODOT_API_CLASS(RemoteAttribute));
	CACHE_CLASS_AND_CHECK(SyncAttribute, GODOT_API_CLASS(SyncAttribute));
	CACHE_CLASS_AND_CHECK(MasterAttribute, GODOT_API_CLASS(MasterAttribute));
	CACHE_CLASS_AND_CHECK(PuppetAttribute, GODOT_API_CLASS(PuppetAttribute));
	CACHE_CLASS_AND_CHECK(SlaveAttribute, GODOT_API_CLASS(SlaveAttribute));
	CACHE_CLASS_AND_CHECK(RemoteSyncAttribute, GODOT_API_CLASS(RemoteSyncAttribute));
	CACHE_CLASS_AND_CHECK(MasterSyncAttribute, GODOT_API_CLASS(MasterSyncAttribute));
	CACHE_CLASS_AND_CHECK(PuppetSyncAttribute, GODOT_API_CLASS(PuppetSyncAttribute));
	CACHE_CLASS_AND_CHECK(GodotMethodAttribute, GODOT_API_CLASS(GodotMethodAttribute));
	CACHE_FIELD_AND_CHECK(GodotMethodAttribute, methodName, CACHED_CLASS(GodotMethodAttribute)->get_field("methodName"));

	CACHE_FIELD_AND_CHECK(GodotObject, ptr, CACHED_CLASS(GodotObject)->get_field(BINDINGS_PTR_FIELD));
	CACHE_FIELD_AND_CHECK(NodePath, ptr, CACHED_CLASS(NodePath)->get_field(BINDINGS_PTR_FIELD));
	CACHE_FIELD_AND_CHECK(RID, ptr, CACHED_CLASS(RID)->get_field(BINDINGS_PTR_FIELD));

	CACHE_METHOD_THUNK_AND_CHECK(GodotObject, Dispose, CACHED_CLASS(GodotObject)->get_method("Dispose", 0));
	CACHE_METHOD_THUNK_AND_CHECK(Array, GetPtr, GODOT_API_NS_CLASS(BINDINGS_NAMESPACE_COLLECTIONS, Array)->get_method("GetPtr", 0));
	CACHE_METHOD_THUNK_AND_CHECK(Dictionary, GetPtr, GODOT_API_NS_CLASS(BINDINGS_NAMESPACE_COLLECTIONS, Dictionary)->get_method("GetPtr", 0));
	CACHE_METHOD_THUNK_AND_CHECK(SignalAwaiter, SignalCallback, GODOT_API_CLASS(SignalAwaiter)->get_method("SignalCallback", 1));
	CACHE_METHOD_THUNK_AND_CHECK(SignalAwaiter, FailureCallback, GODOT_API_CLASS(SignalAwaiter)->get_method("FailureCallback", 0));
	CACHE_METHOD_THUNK_AND_CHECK(GodotTaskScheduler, Activate, GODOT_API_CLASS(GodotTaskScheduler)->get_method("Activate", 0));

	// Start of MarshalUtils methods

	CACHE_METHOD_THUNK_AND_CHECK(MarshalUtils, TypeIsGenericArray, GODOT_API_CLASS(MarshalUtils)->get_method("TypeIsGenericArray", 1));
	CACHE_METHOD_THUNK_AND_CHECK(MarshalUtils, TypeIsGenericDictionary, GODOT_API_CLASS(MarshalUtils)->get_method("TypeIsGenericDictionary", 1));
	CACHE_METHOD_THUNK_AND_CHECK(MarshalUtils, TypeIsSystemGenericList, GODOT_API_CLASS(MarshalUtils)->get_method("TypeIsSystemGenericList", 1));
	CACHE_METHOD_THUNK_AND_CHECK(MarshalUtils, TypeIsSystemGenericDictionary, GODOT_API_CLASS(MarshalUtils)->get_method("TypeIsSystemGenericDictionary", 1));
	CACHE_METHOD_THUNK_AND_CHECK(MarshalUtils, TypeIsGenericIEnumerable, GODOT_API_CLASS(MarshalUtils)->get_method("TypeIsGenericIEnumerable", 1));
	CACHE_METHOD_THUNK_AND_CHECK(MarshalUtils, TypeIsGenericICollection, GODOT_API_CLASS(MarshalUtils)->get_method("TypeIsGenericICollection", 1));
	CACHE_METHOD_THUNK_AND_CHECK(MarshalUtils, TypeIsGenericIDictionary, GODOT_API_CLASS(MarshalUtils)->get_method("TypeIsGenericIDictionary", 1));

	CACHE_METHOD_THUNK_AND_CHECK(MarshalUtils, GetGenericTypeDefinition, GODOT_API_CLASS(MarshalUtils)->get_method("GetGenericTypeDefinition", 2));

	CACHE_METHOD_THUNK_AND_CHECK(MarshalUtils, ArrayGetElementType, GODOT_API_CLASS(MarshalUtils)->get_method("ArrayGetElementType", 2));
	CACHE_METHOD_THUNK_AND_CHECK(MarshalUtils, DictionaryGetKeyValueTypes, GODOT_API_CLASS(MarshalUtils)->get_method("DictionaryGetKeyValueTypes", 3));

	CACHE_METHOD_THUNK_AND_CHECK(MarshalUtils, MakeGenericArrayType, GODOT_API_CLASS(MarshalUtils)->get_method("MakeGenericArrayType", 1));
	CACHE_METHOD_THUNK_AND_CHECK(MarshalUtils, MakeGenericDictionaryType, GODOT_API_CLASS(MarshalUtils)->get_method("MakeGenericDictionaryType", 2));

	// End of MarshalUtils methods

#ifdef DEBUG_ENABLED
	CACHE_METHOD_THUNK_AND_CHECK(DebuggingUtils, GetStackFrameInfo, GODOT_API_CLASS(DebuggingUtils)->get_method("GetStackFrameInfo", 4));
#endif

	// TODO Move to CSharpLanguage::init() and do handle disposal
	MonoObject *task_scheduler = mono_object_new(mono_domain_get(), GODOT_API_CLASS(GodotTaskScheduler)->get_mono_ptr());
	GDMonoUtils::runtime_object_init(task_scheduler, GODOT_API_CLASS(GodotTaskScheduler));
	cached_data.task_scheduler_handle = MonoGCHandle::create_strong(task_scheduler);

	cached_data.godot_api_cache_updated = true;
}

} // namespace GDMonoCache
