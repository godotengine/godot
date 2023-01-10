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
} // namespace GDMonoInternals
