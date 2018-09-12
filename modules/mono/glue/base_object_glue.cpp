/*************************************************************************/
/*  base_object_glue.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "base_object_glue.h"

#ifdef MONO_GLUE_ENABLED

#include "core/reference.h"
#include "core/string_db.h"

#include "../csharp_script.h"
#include "../mono_gd/gd_mono_internals.h"
#include "../mono_gd/gd_mono_utils.h"
#include "../signal_awaiter_utils.h"

Object *godot_icall_Object_Ctor(MonoObject *p_obj) {
	Object *instance = memnew(Object);
	GDMonoInternals::tie_managed_to_unmanaged(p_obj, instance);
	return instance;
}

void godot_icall_Object_Disposed(MonoObject *p_obj, Object *p_ptr) {
#ifdef DEBUG_ENABLED
	CRASH_COND(p_ptr == NULL);
#endif

	if (p_ptr->get_script_instance()) {
		CSharpInstance *cs_instance = CAST_CSHARP_INSTANCE(p_ptr->get_script_instance());
		if (cs_instance) {
			cs_instance->mono_object_disposed(p_obj);
			p_ptr->set_script_instance(NULL);
			return;
		}
	}

	void *data = p_ptr->get_script_instance_binding(CSharpLanguage::get_singleton()->get_language_index());

	if (data) {
		Ref<MonoGCHandle> &gchandle = ((Map<Object *, CSharpScriptBinding>::Element *)data)->get().gchandle;
		if (gchandle.is_valid()) {
			CSharpLanguage::release_script_gchandle(p_obj, gchandle);
		}
	}
}

void godot_icall_Reference_Disposed(MonoObject *p_obj, Object *p_ptr, bool p_is_finalizer) {
#ifdef DEBUG_ENABLED
	CRASH_COND(p_ptr == NULL);
	// This is only called with Reference derived classes
	CRASH_COND(!Object::cast_to<Reference>(p_ptr));
#endif

	Reference *ref = static_cast<Reference *>(p_ptr);

	if (ref->get_script_instance()) {
		CSharpInstance *cs_instance = CAST_CSHARP_INSTANCE(ref->get_script_instance());
		if (cs_instance) {
			bool r_owner_deleted;
			cs_instance->mono_object_disposed_baseref(p_obj, p_is_finalizer, r_owner_deleted);
			if (!r_owner_deleted && !p_is_finalizer) {
				// If the native instance is still alive and Dispose() was called
				// (instead of the finalizer), then we remove the script instance.
				ref->set_script_instance(NULL);
			}
			return;
		}
	}

	// Unsafe refcount decrement. The managed instance also counts as a reference.
	// See: CSharpLanguage::alloc_instance_binding_data(Object *p_object)
	if (ref->unreference()) {
		memdelete(ref);
	} else {
		void *data = ref->get_script_instance_binding(CSharpLanguage::get_singleton()->get_language_index());

		if (data) {
			Ref<MonoGCHandle> &gchandle = ((Map<Object *, CSharpScriptBinding>::Element *)data)->get().gchandle;
			if (gchandle.is_valid()) {
				CSharpLanguage::release_script_gchandle(p_obj, gchandle);
			}
		}
	}
}

MethodBind *godot_icall_Object_ClassDB_get_method(MonoString *p_type, MonoString *p_method) {
	StringName type(GDMonoMarshal::mono_string_to_godot(p_type));
	StringName method(GDMonoMarshal::mono_string_to_godot(p_method));
	return ClassDB::get_method(type, method);
}

MonoObject *godot_icall_Object_weakref(Object *p_obj) {
	if (!p_obj)
		return NULL;

	Ref<WeakRef> wref;
	Reference *ref = Object::cast_to<Reference>(p_obj);

	if (ref) {
		REF r = ref;
		if (!r.is_valid())
			return NULL;

		wref.instance();
		wref->set_ref(r);
	} else {
		wref.instance();
		wref->set_obj(p_obj);
	}

	return GDMonoUtils::create_managed_for_godot_object(CACHED_CLASS(WeakRef), Reference::get_class_static(), Object::cast_to<Object>(wref.ptr()));
}

Error godot_icall_SignalAwaiter_connect(Object *p_source, MonoString *p_signal, Object *p_target, MonoObject *p_awaiter) {
	String signal = GDMonoMarshal::mono_string_to_godot(p_signal);
	return SignalAwaiterUtils::connect_signal_awaiter(p_source, signal, p_target, p_awaiter);
}

void godot_register_object_icalls() {
	mono_add_internal_call("Godot.Object::godot_icall_Object_Ctor", (void *)godot_icall_Object_Ctor);
	mono_add_internal_call("Godot.Object::godot_icall_Object_Disposed", (void *)godot_icall_Object_Disposed);
	mono_add_internal_call("Godot.Object::godot_icall_Reference_Disposed", (void *)godot_icall_Reference_Disposed);
	mono_add_internal_call("Godot.Object::godot_icall_Object_ClassDB_get_method", (void *)godot_icall_Object_ClassDB_get_method);
	mono_add_internal_call("Godot.Object::godot_icall_Object_weakref", (void *)godot_icall_Object_weakref);
	mono_add_internal_call("Godot.SignalAwaiter::godot_icall_SignalAwaiter_connect", (void *)godot_icall_SignalAwaiter_connect);
}

#endif // MONO_GLUE_ENABLED
