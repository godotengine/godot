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
	GDMonoClass *class_MonoObject; // object
	GDMonoClass *class_bool; // bool
	GDMonoClass *class_int8_t; // sbyte
	GDMonoClass *class_int16_t; // short
	GDMonoClass *class_int32_t; // int
	GDMonoClass *class_int64_t; // long
	GDMonoClass *class_uint8_t; // byte
	GDMonoClass *class_uint16_t; // ushort
	GDMonoClass *class_uint32_t; // uint
	GDMonoClass *class_uint64_t; // ulong
	GDMonoClass *class_float; // float
	GDMonoClass *class_double; // double
	GDMonoClass *class_String; // string
	GDMonoClass *class_IntPtr; // System.IntPtr

	GDMonoClass *class_System_Collections_IEnumerable;
	GDMonoClass *class_System_Collections_ICollection;
	GDMonoClass *class_System_Collections_IDictionary;

#ifdef DEBUG_ENABLED
	GDMonoClass *class_System_Diagnostics_StackTrace;
	GDMonoMethodThunkR<MonoArray *, MonoObject *> methodthunk_System_Diagnostics_StackTrace_GetFrames;
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
	GDMonoMethodThunk<MonoObject *, MonoString **, int *, MonoString **> methodthunk_DebuggingUtils_GetStackFrameInfo;
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

	GDMonoMethodThunk<MonoObject *> methodthunk_GodotObject_Dispose;
	GDMonoMethodThunkR<Array *, MonoObject *> methodthunk_Array_GetPtr;
	GDMonoMethodThunkR<Dictionary *, MonoObject *> methodthunk_Dictionary_GetPtr;
	GDMonoMethodThunk<MonoObject *, MonoArray *> methodthunk_SignalAwaiter_SignalCallback;
	GDMonoMethodThunk<MonoObject *> methodthunk_SignalAwaiter_FailureCallback;
	GDMonoMethodThunk<MonoObject *> methodthunk_GodotTaskScheduler_Activate;

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

	Ref<MonoGCHandle> task_scheduler_handle;

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

#ifdef REAL_T_IS_DOUBLE
#define REAL_T_MONOCLASS CACHED_CLASS_RAW(double)
#else
#define REAL_T_MONOCLASS CACHED_CLASS_RAW(float)
#endif

#endif // GD_MONO_CACHE_H
