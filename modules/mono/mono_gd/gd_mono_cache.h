/*************************************************************************/
/*  gd_mono_cache.h                                                      */
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
	GDMonoClass *class_String; // string

#ifdef DEBUG_ENABLED
	GDMonoClass *class_System_Diagnostics_StackTrace;
	GDMonoMethodThunkR<MonoArray *, MonoObject *> methodthunk_System_Diagnostics_StackTrace_GetFrames;
	GDMonoMethod *method_System_Diagnostics_StackTrace_ctor_bool;
	GDMonoMethod *method_System_Diagnostics_StackTrace_ctor_Exception_bool;
#endif

	GDMonoClass *class_KeyNotFoundException;
	// -----------------------------------------------

	GDMonoClass *class_GodotObject;
	GDMonoClass *class_GodotResource;
	GDMonoClass *class_Control;
	GDMonoClass *class_Callable;
	GDMonoClass *class_SignalInfo;
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
	GDMonoClass *class_MasterAttribute;
	GDMonoClass *class_PuppetAttribute;
	GDMonoClass *class_GodotMethodAttribute;
	GDMonoField *field_GodotMethodAttribute_methodName;
	GDMonoClass *class_ScriptPathAttribute;
	GDMonoField *field_ScriptPathAttribute_path;
	GDMonoClass *class_AssemblyHasScriptsAttribute;
	GDMonoField *field_AssemblyHasScriptsAttribute_requiresLookup;
	GDMonoField *field_AssemblyHasScriptsAttribute_scriptTypes;

	GDMonoField *field_GodotObject_ptr;

	GDMonoMethodThunk<MonoObject *> methodthunk_GodotObject_Dispose;
	GDMonoMethodThunk<MonoObject *, MonoArray *> methodthunk_SignalAwaiter_SignalCallback;
	GDMonoMethodThunk<MonoObject *> methodthunk_GodotTaskScheduler_Activate;

	GDMonoMethodThunkR<MonoBoolean, MonoObject *, MonoObject *> methodthunk_Delegate_Equals;

	GDMonoMethodThunkR<MonoBoolean, void *, MonoObject *> methodthunk_DelegateUtils_TrySerializeDelegateWithGCHandle;
	GDMonoMethodThunkR<MonoBoolean, MonoObject *, void **> methodthunk_DelegateUtils_TryDeserializeDelegateWithGCHandle;

	GDMonoMethodThunkR<MonoBoolean, MonoDelegate *, MonoObject *> methodthunk_DelegateUtils_TrySerializeDelegate;
	GDMonoMethodThunkR<MonoBoolean, MonoObject *, MonoDelegate **> methodthunk_DelegateUtils_TryDeserializeDelegate;

	GDMonoMethodThunk<void *, const Variant **, uint32_t, const Variant *> methodthunk_DelegateUtils_InvokeWithVariantArgs;
	GDMonoMethodThunkR<MonoBoolean, void *, void *> methodthunk_DelegateUtils_DelegateEquals;
	GDMonoMethodThunk<void *> methodthunk_DelegateUtils_FreeGCHandle;

	GDMonoMethodThunkR<int32_t, MonoReflectionType *, MonoBoolean *> methodthunk_Marshaling_managed_to_variant_type;
	GDMonoMethodThunkR<MonoBoolean, MonoReflectionType *, MonoReflectionType **> methodthunk_Marshaling_try_get_array_element_type;
	GDMonoMethodThunkR<MonoObject *, const Variant *, MonoReflectionType *> methodthunk_Marshaling_variant_to_mono_object_of_type;
	GDMonoMethodThunkR<MonoObject *, const Variant *> methodthunk_Marshaling_variant_to_mono_object;
	GDMonoMethodThunk<MonoObject *, MonoBoolean, Variant *> methodthunk_Marshaling_mono_object_to_variant_out;

	GDMonoMethodThunk<MonoReflectionField *, MonoObject *, const Variant *> methodthunk_Marshaling_SetFieldValue;

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

inline void clear_godot_api_cache() {
	cached_data.clear_godot_api_cache();
}
} // namespace GDMonoCache

#define CACHED_CLASS(m_class) (GDMonoCache::cached_data.class_##m_class)
#define CACHED_CLASS_RAW(m_class) (GDMonoCache::cached_data.class_##m_class->get_mono_ptr())
#define CACHED_FIELD(m_class, m_field) (GDMonoCache::cached_data.field_##m_class##_##m_field)
#define CACHED_METHOD(m_class, m_method) (GDMonoCache::cached_data.method_##m_class##_##m_method)
#define CACHED_METHOD_THUNK(m_class, m_method) (GDMonoCache::cached_data.methodthunk_##m_class##_##m_method)

#endif // GD_MONO_CACHE_H
