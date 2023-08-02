/**************************************************************************/
/*  gd_mono_internals.cpp                                                 */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "gd_mono_internals.h"

#include "../csharp_script.h"
#include "../mono_gc_handle.h"
#include "../utils/macros.h"
#include "../utils/thread_local.h"
#include "gd_mono_class.h"
#include "gd_mono_marshal.h"
#include "gd_mono_utils.h"

#include <mono/metadata/exception.h>

namespace GDMonoInternals {
void tie_managed_to_unmanaged(MonoObject *managed, Object *unmanaged) {
	// This method should not fail

	CRASH_COND(!unmanaged);

	// All mono objects created from the managed world (e.g.: 'new Player()')
	// need to have a CSharpScript in order for their methods to be callable from the unmanaged side

	Reference *ref = Object::cast_to<Reference>(unmanaged);

	GDMonoClass *klass = GDMonoUtils::get_object_class(managed);

	CRASH_COND(!klass);

	GDMonoClass *native = GDMonoUtils::get_class_native_base(klass);

	CRASH_COND(native == NULL);

	if (native == klass) {
		// If it's just a wrapper Godot class and not a custom inheriting class, then attach a
		// script binding instead. One of the advantages of this is that if a script is attached
		// later and it's not a C# script, then the managed object won't have to be disposed.
		// Another reason for doing this is that this instance could outlive CSharpLanguage, which would
		// be problematic when using a script. See: https://github.com/godotengine/godot/issues/25621

		CSharpScriptBinding script_binding;

		script_binding.inited = true;
		script_binding.type_name = NATIVE_GDMONOCLASS_NAME(klass);
		script_binding.wrapper_class = klass;
		script_binding.gchandle = ref ? MonoGCHandle::create_weak(managed) : MonoGCHandle::create_strong(managed);
		script_binding.owner = unmanaged;

		if (ref) {
			// Unsafe refcount increment. The managed instance also counts as a reference.
			// This way if the unmanaged world has no references to our owner
			// but the managed instance is alive, the refcount will be 1 instead of 0.
			// See: godot_icall_Reference_Dtor(MonoObject *p_obj, Object *p_ptr)

			// May not me referenced yet, so we must use init_ref() instead of reference()
			if (ref->init_ref()) {
				CSharpLanguage::get_singleton()->post_unsafe_reference(ref);
			}
		}

		// The object was just created, no script instance binding should have been attached
		CRASH_COND(unmanaged->has_script_instance_binding(CSharpLanguage::get_singleton()->get_language_index()));

		void *data;
		{
			MutexLock lock(CSharpLanguage::get_singleton()->get_language_bind_mutex());
			data = (void *)CSharpLanguage::get_singleton()->insert_script_binding(unmanaged, script_binding);
		}

		// Should be thread safe because the object was just created and nothing else should be referencing it
		unmanaged->set_script_instance_binding(CSharpLanguage::get_singleton()->get_language_index(), data);

		return;
	}

	Ref<MonoGCHandle> gchandle = ref ? MonoGCHandle::create_weak(managed) : MonoGCHandle::create_strong(managed);

	Ref<CSharpScript> script = CSharpScript::create_for_managed_type(klass, native);

	CRASH_COND(script.is_null());

	ScriptInstance *si = CSharpInstance::create_for_managed_type(unmanaged, script.ptr(), gchandle);

	unmanaged->set_script_and_instance(script.get_ref_ptr(), si);
}

void unhandled_exception(MonoException *p_exc) {
	mono_print_unhandled_exception((MonoObject *)p_exc);
	gd_unhandled_exception_event(p_exc);

	if (GDMono::get_singleton()->get_unhandled_exception_policy() == GDMono::POLICY_TERMINATE_APP) {
		// Too bad 'mono_invoke_unhandled_exception_hook' is not exposed to embedders
		mono_unhandled_exception((MonoObject *)p_exc);
		GDMono::unhandled_exception_hook((MonoObject *)p_exc, NULL);
		GD_UNREACHABLE();
	} else {
#ifdef DEBUG_ENABLED
		GDMonoUtils::debug_send_unhandled_exception_error((MonoException *)p_exc);
		if (ScriptDebugger::get_singleton())
			ScriptDebugger::get_singleton()->idle_poll();
#endif
	}
}

void gd_unhandled_exception_event(MonoException *p_exc) {
	MonoImage *mono_image = GDMono::get_singleton()->get_core_api_assembly()->get_image();

	MonoClass *gd_klass = mono_class_from_name(mono_image, "Godot", "GD");
	MonoMethod *unhandled_exception_method = mono_class_get_method_from_name(gd_klass, "OnUnhandledException", -1);
	void *args[1];
	args[0] = p_exc;
	mono_runtime_invoke(unhandled_exception_method, nullptr, (void **)args, nullptr);
}

#if DEBUG_ENABLED
static String _get_var_type(const Variant *p_var) {
	String basestr;

	if (p_var->get_type() == Variant::OBJECT) {
		Object *bobj = *p_var;
		if (!bobj) {
			if (p_var->is_invalid_object()) {
				basestr = "previously freed instance";
			} else {
				basestr = "null instance";
			}
		} else {
			if (bobj->get_script_instance()) {
				basestr = bobj->get_class() + " (" + bobj->get_script_instance()->get_script()->get_path().get_file() + ")";
			} else {
				basestr = bobj->get_class();
			}
		}

	} else {
		basestr = Variant::get_type_name(p_var->get_type());
	}

	return basestr;
}

static String _get_call_where(const String &p_method, const Variant *p_instance, const Variant **argptrs, int argc) {
	String methodstr = p_method;
	String basestr = _get_var_type(p_instance);

	if (methodstr == "call") {
		if (argc >= 1) {
			methodstr = String(*argptrs[0]) + " (via call)";
		}
	} else if (methodstr == "call_recursive" && basestr == "TreeItem") {
		if (argc >= 1) {
			methodstr = String(*argptrs[0]) + " (via TreeItem.call_recursive)";
		}
	}
	return "function '" + methodstr + "' in base '" + basestr + "'";
}

static String _get_call_error(const Variant::CallError &p_err, const String &p_where, const Variant **argptrs) {
	String err_text;

	if (p_err.error == Variant::CallError::CALL_ERROR_INVALID_ARGUMENT) {
		int errorarg = p_err.argument;
		// Handle the Object to Object case separately as we don't have further class details.
#ifdef DEBUG_ENABLED
		if (p_err.expected == Variant::OBJECT && argptrs[errorarg]->get_type() == p_err.expected) {
			err_text = "Invalid type in " + p_where + ". The Object-derived class of argument " + itos(errorarg + 1) + " (" + _get_var_type(argptrs[errorarg]) + ") is not a subclass of the expected argument class.";
		} else
#endif // DEBUG_ENABLED
		{
			err_text = "Invalid type in " + p_where + ". Cannot convert argument " + itos(errorarg + 1) + " from " + Variant::get_type_name(argptrs[errorarg]->get_type()) + " to " + Variant::get_type_name(p_err.expected) + ".";
		}
	} else if (p_err.error == Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS) {
		err_text = "Invalid call to " + p_where + ". Expected " + itos(p_err.argument) + " arguments.";
	} else if (p_err.error == Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS) {
		err_text = "Invalid call to " + p_where + ". Expected " + itos(p_err.argument) + " arguments.";
	} else if (p_err.error == Variant::CallError::CALL_ERROR_INVALID_METHOD) {
		err_text = "Invalid call. Nonexistent " + p_where + ".";
	} else if (p_err.error == Variant::CallError::CALL_ERROR_INSTANCE_IS_NULL) {
		err_text = "Attempt to call " + p_where + " on a null instance.";
	} else {
		err_text = "Bug, call error: #" + itos(p_err.error);
	}

	return err_text;
}

void check_call_error(const String &p_method, const Variant *p_instance, const Variant **p_args, int p_arg_count, const Variant::CallError &p_error) {
	if (p_error.error == Variant::CallError::CALL_OK) {
		// The call was successful.
		return;
	}

	const String &where = _get_call_where(p_method, p_instance, p_args, p_arg_count);
	ERR_PRINT(_get_call_error(p_error, where, p_args));
}
#else
void check_call_error(const String &p_method, const Variant &p_instance, const Variant **p_args, int p_arg_count, const Variant::CallError &p_error) {}
#endif
} // namespace GDMonoInternals
