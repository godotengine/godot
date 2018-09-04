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

#include "../mono_gd/gd_mono_internals.h"
#include "../mono_gd/gd_mono_utils.h"
#include "../signal_awaiter_utils.h"

Object *godot_icall_Object_Ctor(MonoObject *obj) {
	Object *instance = memnew(Object);
	GDMonoInternals::tie_managed_to_unmanaged(obj, instance);
	return instance;
}

void godot_icall_Object_Dtor(MonoObject *obj, Object *ptr) {
#ifdef DEBUG_ENABLED
	CRASH_COND(ptr == NULL);
#endif
	_GodotSharp::get_singleton()->queue_dispose(obj, ptr);
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
	mono_add_internal_call("Godot.Object::godot_icall_Object_Dtor", (void *)godot_icall_Object_Dtor);
	mono_add_internal_call("Godot.Object::godot_icall_Object_ClassDB_get_method", (void *)godot_icall_Object_ClassDB_get_method);
	mono_add_internal_call("Godot.Object::godot_icall_Object_weakref", (void *)godot_icall_Object_weakref);
	mono_add_internal_call("Godot.SignalAwaiter::godot_icall_SignalAwaiter_connect", (void *)godot_icall_SignalAwaiter_connect);
}

#endif // MONO_GLUE_ENABLED
