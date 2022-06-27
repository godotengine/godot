/*************************************************************************/
/*  gd_mono_cache.h                                                      */
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

#ifndef GD_MONO_CACHE_H
#define GD_MONO_CACHE_H

#include "gd_mono_header.h"
#include "gd_mono_method_thunk.h"

namespace GDMonoCache {

struct CachedData {
	// -----------------------------------------------
	// corlib classes

	// Let's use the no-namespace format for these too
	GDMonoClass *class_MonoObject = nullptr; // object
	GDMonoClass *class_bool = nullptr; // bool
	GDMonoClass *class_int8_t = nullptr; // sbyte
	GDMonoClass *class_int16_t = nullptr; // short
	GDMonoClass *class_int32_t = nullptr; // int
	GDMonoClass *class_int64_t = nullptr; // long
	GDMonoClass *class_uint8_t = nullptr; // byte
	GDMonoClass *class_uint16_t = nullptr; // ushort
	GDMonoClass *class_uint32_t = nullptr; // uint
	GDMonoClass *class_uint64_t = nullptr; // ulong
	GDMonoClass *class_float = nullptr; // float
	GDMonoClass *class_double = nullptr; // double
	GDMonoClass *class_String = nullptr; // string
	GDMonoClass *class_IntPtr = nullptr; // System.IntPtr

	GDMonoClass *class_System_Collections_IEnumerable = nullptr;
	GDMonoClass *class_System_Collections_ICollection = nullptr;
	GDMonoClass *class_System_Collections_IDictionary = nullptr;

#ifdef DEBUG_ENABLED
	GDMonoClass *class_System_Diagnostics_StackTrace = nullptr;
	GDMonoMethodThunkR<MonoArray *, MonoObject *> methodthunk_System_Diagnostics_StackTrace_GetFrames;
	GDMonoMethod *method_System_Diagnostics_StackTrace_ctor_bool = nullptr;
	GDMonoMethod *method_System_Diagnostics_StackTrace_ctor_Exception_bool = nullptr;
#endif

	GDMonoClass *class_KeyNotFoundException = nullptr;

	MonoClass *rawclass_Dictionary = nullptr;
	// -----------------------------------------------

	GDMonoClass *class_Vector2 = nullptr;
	GDMonoClass *class_Vector2i = nullptr;
	GDMonoClass *class_Rect2 = nullptr;
	GDMonoClass *class_Rect2i = nullptr;
	GDMonoClass *class_Transform2D = nullptr;
	GDMonoClass *class_Vector3 = nullptr;
	GDMonoClass *class_Vector3i = nullptr;
	GDMonoClass *class_Basis = nullptr;
	GDMonoClass *class_Quaternion = nullptr;
	GDMonoClass *class_Transform3D = nullptr;
	GDMonoClass *class_AABB = nullptr;
	GDMonoClass *class_Color = nullptr;
	GDMonoClass *class_Plane = nullptr;
	GDMonoClass *class_StringName = nullptr;
	GDMonoClass *class_NodePath = nullptr;
	GDMonoClass *class_RID = nullptr;
	GDMonoClass *class_GodotObject = nullptr;
	GDMonoClass *class_GodotResource = nullptr;
	GDMonoClass *class_Node = nullptr;
	GDMonoClass *class_Control = nullptr;
	GDMonoClass *class_Node3D = nullptr;
	GDMonoClass *class_WeakRef = nullptr;
	GDMonoClass *class_Callable = nullptr;
	GDMonoClass *class_SignalInfo = nullptr;
	GDMonoClass *class_Array = nullptr;
	GDMonoClass *class_Dictionary = nullptr;
	GDMonoClass *class_MarshalUtils = nullptr;
	GDMonoClass *class_ISerializationListener = nullptr;

#ifdef DEBUG_ENABLED
	GDMonoClass *class_DebuggingUtils = nullptr;
	GDMonoMethodThunk<MonoObject *, MonoString **, int *, MonoString **> methodthunk_DebuggingUtils_GetStackFrameInfo;
#endif

	GDMonoClass *class_ExportAttribute = nullptr;
	GDMonoField *field_ExportAttribute_hint = nullptr;
	GDMonoField *field_ExportAttribute_hintString = nullptr;
	GDMonoClass *class_SignalAttribute = nullptr;
	GDMonoClass *class_ToolAttribute = nullptr;
	GDMonoClass *class_AnyPeerAttribute = nullptr;
	GDMonoClass *class_AuthorityAttribute = nullptr;
	GDMonoClass *class_GodotMethodAttribute = nullptr;
	GDMonoField *field_GodotMethodAttribute_methodName = nullptr;
	GDMonoClass *class_ScriptPathAttribute = nullptr;
	GDMonoField *field_ScriptPathAttribute_path = nullptr;
	GDMonoClass *class_AssemblyHasScriptsAttribute = nullptr;
	GDMonoField *field_AssemblyHasScriptsAttribute_requiresLookup = nullptr;
	GDMonoField *field_AssemblyHasScriptsAttribute_scriptTypes = nullptr;

	GDMonoField *field_GodotObject_ptr = nullptr;
	GDMonoField *field_StringName_ptr = nullptr;
	GDMonoField *field_NodePath_ptr = nullptr;
	GDMonoField *field_Image_ptr = nullptr;
	GDMonoField *field_RID_ptr = nullptr;

	GDMonoMethodThunk<MonoObject *> methodthunk_GodotObject_Dispose;
	GDMonoMethodThunkR<Array *, MonoObject *> methodthunk_Array_GetPtr;
	GDMonoMethodThunkR<Dictionary *, MonoObject *> methodthunk_Dictionary_GetPtr;
	GDMonoMethodThunk<MonoObject *, MonoArray *> methodthunk_SignalAwaiter_SignalCallback;
	GDMonoMethodThunk<MonoObject *> methodthunk_GodotTaskScheduler_Activate;

	GDMonoMethodThunkR<MonoBoolean, MonoObject *, MonoObject *> methodthunk_Delegate_Equals;

	GDMonoMethodThunkR<MonoBoolean, MonoDelegate *, MonoObject *> methodthunk_DelegateUtils_TrySerializeDelegate;
	GDMonoMethodThunkR<MonoBoolean, MonoObject *, MonoDelegate **> methodthunk_DelegateUtils_TryDeserializeDelegate;

	// Start of MarshalUtils methods

	GDMonoMethodThunkR<MonoBoolean, MonoReflectionType *> methodthunk_MarshalUtils_TypeIsGenericArray;
	GDMonoMethodThunkR<MonoBoolean, MonoReflectionType *> methodthunk_MarshalUtils_TypeIsGenericDictionary;
	GDMonoMethodThunkR<MonoBoolean, MonoReflectionType *> methodthunk_MarshalUtils_TypeIsSystemGenericList;
	GDMonoMethodThunkR<MonoBoolean, MonoReflectionType *> methodthunk_MarshalUtils_TypeIsSystemGenericDictionary;
	GDMonoMethodThunkR<MonoBoolean, MonoReflectionType *> methodthunk_MarshalUtils_TypeIsGenericIEnumerable;
	GDMonoMethodThunkR<MonoBoolean, MonoReflectionType *> methodthunk_MarshalUtils_TypeIsGenericICollection;
	GDMonoMethodThunkR<MonoBoolean, MonoReflectionType *> methodthunk_MarshalUtils_TypeIsGenericIDictionary;
	GDMonoMethodThunkR<MonoBoolean, MonoReflectionType *> methodthunk_MarshalUtils_TypeHasFlagsAttribute;

	GDMonoMethodThunk<MonoReflectionType *, MonoReflectionType **> methodthunk_MarshalUtils_GetGenericTypeDefinition;

	GDMonoMethodThunk<MonoReflectionType *, MonoReflectionType **> methodthunk_MarshalUtils_ArrayGetElementType;
	GDMonoMethodThunk<MonoReflectionType *, MonoReflectionType **, MonoReflectionType **> methodthunk_MarshalUtils_DictionaryGetKeyValueTypes;

	GDMonoMethodThunkR<MonoReflectionType *, MonoReflectionType *> methodthunk_MarshalUtils_MakeGenericArrayType;
	GDMonoMethodThunkR<MonoReflectionType *, MonoReflectionType *, MonoReflectionType *> methodthunk_MarshalUtils_MakeGenericDictionaryType;

	// End of MarshalUtils methods

	Ref<MonoGCHandleRef> task_scheduler_handle;

	bool corlib_cache_updated;
	bool godot_api_cache_updated;

	void clear_corlib_cache();
	void clear_godot_api_cache();

	CachedData() {
		clear_corlib_cache();
		clear_godot_api_cache();
	}
};

extern CachedData cached_data;

void update_corlib_cache();
void update_godot_api_cache();

inline void clear_corlib_cache() {
	cached_data.clear_corlib_cache();
}

inline void clear_godot_api_cache() {
	cached_data.clear_godot_api_cache();
}
} // namespace GDMonoCache

#define CACHED_CLASS(m_class) (GDMonoCache::cached_data.class_##m_class)
#define CACHED_CLASS_RAW(m_class) (GDMonoCache::cached_data.class_##m_class->get_mono_ptr())
#define CACHED_RAW_MONO_CLASS(m_class) (GDMonoCache::cached_data.rawclass_##m_class)
#define CACHED_FIELD(m_class, m_field) (GDMonoCache::cached_data.field_##m_class##_##m_field)
#define CACHED_METHOD(m_class, m_method) (GDMonoCache::cached_data.method_##m_class##_##m_method)
#define CACHED_METHOD_THUNK(m_class, m_method) (GDMonoCache::cached_data.methodthunk_##m_class##_##m_method)
#define CACHED_PROPERTY(m_class, m_property) (GDMonoCache::cached_data.property_##m_class##_##m_property)

#endif // GD_MONO_CACHE_H
