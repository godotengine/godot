/*************************************************************************/
/*  placeholder_glue.cpp                                                 */
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

#ifndef GLUE_HEADER_H
#define GLUE_HEADER_H

#include "core/object/object.h"

#include "../csharp_script.h"
#include "../mono_gd/gd_mono_cache.h"
#include "../mono_gd/gd_mono_internals.h"
#include "../mono_gd/gd_mono_utils.h"

GCHandleIntPtr unmanaged_get_script_instance_managed(Object *p_unmanaged, bool *r_has_cs_script_instance) {
#ifdef DEBUG_ENABLED
	CRASH_COND(!p_unmanaged);
	CRASH_COND(!r_has_cs_script_instance);
#endif

	if (p_unmanaged->get_script_instance()) {
		CSharpInstance *cs_instance = CAST_CSHARP_INSTANCE(p_unmanaged->get_script_instance());

		if (cs_instance) {
			*r_has_cs_script_instance = true;
			return cs_instance->get_gchandle_intptr();
		}
	}

	*r_has_cs_script_instance = false;
	return GCHandleIntPtr();
}

GCHandleIntPtr unmanaged_get_instance_binding_managed(Object *p_unmanaged) {
#ifdef DEBUG_ENABLED
	CRASH_COND(!p_unmanaged);
#endif

	void *data = CSharpLanguage::get_instance_binding(p_unmanaged);
	ERR_FAIL_NULL_V(data, GCHandleIntPtr());
	CSharpScriptBinding &script_binding = ((RBMap<Object *, CSharpScriptBinding>::Element *)data)->value();
	ERR_FAIL_COND_V(!script_binding.inited, GCHandleIntPtr());

	return script_binding.gchandle.get_intptr();
}

GCHandleIntPtr unmanaged_instance_binding_create_managed(Object *p_unmanaged, GCHandleIntPtr p_old_gchandle) {
#ifdef DEBUG_ENABLED
	CRASH_COND(!p_unmanaged);
#endif

	void *data = CSharpLanguage::get_instance_binding(p_unmanaged);
	ERR_FAIL_NULL_V(data, GCHandleIntPtr());
	CSharpScriptBinding &script_binding = ((RBMap<Object *, CSharpScriptBinding>::Element *)data)->value();
	ERR_FAIL_COND_V(!script_binding.inited, GCHandleIntPtr());

	MonoGCHandleData &gchandle = script_binding.gchandle;

	// TODO: Possible data race?
	CRASH_COND(gchandle.get_intptr().value != p_old_gchandle.value);

	CSharpLanguage::get_singleton()->release_script_gchandle(gchandle);
	script_binding.inited = false;

	// Create a new one

#ifdef DEBUG_ENABLED
	CRASH_COND(script_binding.type_name == StringName());
#endif

	bool parent_is_object_class = ClassDB::is_parent_class(p_unmanaged->get_class_name(), script_binding.type_name);
	ERR_FAIL_COND_V_MSG(!parent_is_object_class, GCHandleIntPtr(),
			"Type inherits from native type '" + script_binding.type_name + "', so it can't be instantiated in object of type: '" + p_unmanaged->get_class() + "'.");

	MonoException *exc = nullptr;
	GCHandleIntPtr strong_gchandle =
			GDMonoCache::cached_data.methodthunk_ScriptManagerBridge_CreateManagedForGodotObjectBinding
					.invoke(&script_binding.type_name, p_unmanaged, &exc);

	if (exc) {
		GDMonoUtils::set_pending_exception(exc);
		return GCHandleIntPtr();
	}

	ERR_FAIL_NULL_V(strong_gchandle.value, GCHandleIntPtr());

	gchandle = MonoGCHandleData(strong_gchandle, gdmono::GCHandleType::STRONG_HANDLE);
	script_binding.inited = true;

	// Tie managed to unmanaged
	RefCounted *rc = Object::cast_to<RefCounted>(p_unmanaged);

	if (rc) {
		// Unsafe refcount increment. The managed instance also counts as a reference.
		// This way if the unmanaged world has no references to our owner
		// but the managed instance is alive, the refcount will be 1 instead of 0.
		// See: godot_icall_RefCounted_Dtor(MonoObject *p_obj, Object *p_ptr)
		rc->reference();
		CSharpLanguage::get_singleton()->post_unsafe_reference(rc);
	}

	return gchandle.get_intptr();
}

void godot_icall_InteropUtils_tie_native_managed_to_unmanaged(GCHandleIntPtr p_gchandle_intptr, Object *p_unmanaged, const StringName *p_native_name, bool p_ref_counted) {
	CSharpLanguage::tie_native_managed_to_unmanaged(p_gchandle_intptr, p_unmanaged, p_native_name, p_ref_counted);
}

void godot_icall_InteropUtils_tie_user_managed_to_unmanaged(GCHandleIntPtr p_gchandle_intptr, Object *p_unmanaged, CSharpScript *p_script, bool p_ref_counted) {
	CSharpLanguage::tie_user_managed_to_unmanaged(p_gchandle_intptr, p_unmanaged, p_script, p_ref_counted);
}

void godot_icall_InteropUtils_tie_managed_to_unmanaged_with_pre_setup(GCHandleIntPtr p_gchandle_intptr, Object *p_unmanaged) {
	CSharpLanguage::tie_managed_to_unmanaged_with_pre_setup(p_gchandle_intptr, p_unmanaged);
}

CSharpScript *godot_icall_InteropUtils_internal_new_csharp_script() {
	CSharpScript *script = memnew(CSharpScript);
	CRASH_COND(!script);
	return script;
}

void godotsharp_array_filter_godot_objects_by_native(StringName *p_native_name, const Array *p_input, Array *r_output) {
	memnew_placement(r_output, Array);

	for (int i = 0; i < p_input->size(); ++i) {
		if (ClassDB::is_parent_class(((Object *)(*p_input)[i])->get_class(), *p_native_name)) {
			r_output->push_back(p_input[i]);
		}
	}
}

void godotsharp_array_filter_godot_objects_by_non_native(const Array *p_input, Array *r_output) {
	memnew_placement(r_output, Array);

	for (int i = 0; i < p_input->size(); ++i) {
		CSharpInstance *si = CAST_CSHARP_INSTANCE(((Object *)(*p_input)[i])->get_script_instance());

		if (si != nullptr) {
			r_output->push_back(p_input[i]);
		}
	}
}

void godot_register_placeholder_icalls() {
	GDMonoUtils::add_internal_call(
			"Godot.NativeInterop.InteropUtils::unmanaged_get_script_instance_managed",
			unmanaged_get_script_instance_managed);
	GDMonoUtils::add_internal_call(
			"Godot.NativeInterop.InteropUtils::unmanaged_get_instance_binding_managed",
			unmanaged_get_instance_binding_managed);
	GDMonoUtils::add_internal_call(
			"Godot.NativeInterop.InteropUtils::unmanaged_instance_binding_create_managed",
			unmanaged_instance_binding_create_managed);
	GDMonoUtils::add_internal_call(
			"Godot.NativeInterop.InteropUtils::internal_tie_native_managed_to_unmanaged",
			godot_icall_InteropUtils_tie_native_managed_to_unmanaged);
	GDMonoUtils::add_internal_call(
			"Godot.NativeInterop.InteropUtils::internal_tie_user_managed_to_unmanaged",
			godot_icall_InteropUtils_tie_user_managed_to_unmanaged);
	GDMonoUtils::add_internal_call(
			"Godot.NativeInterop.InteropUtils::internal_tie_managed_to_unmanaged_with_pre_setup",
			godot_icall_InteropUtils_tie_managed_to_unmanaged_with_pre_setup);
	GDMonoUtils::add_internal_call(
			"Godot.NativeInterop.InteropUtils::internal_new_csharp_script",
			godot_icall_InteropUtils_internal_new_csharp_script);
	GDMonoUtils::add_internal_call(
			"Godot.NativeInterop.SceneTree::godotsharp_array_filter_godot_objects_by_native",
			godotsharp_array_filter_godot_objects_by_native);
	GDMonoUtils::add_internal_call(
			"Godot.NativeInterop.SceneTree::godotsharp_array_filter_godot_objects_by_non_native",
			godotsharp_array_filter_godot_objects_by_non_native);
}

#endif // GLUE_HEADER_H
