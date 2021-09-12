/*************************************************************************/
/*  gd_mono_internals.cpp                                                */
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

#include "gd_mono_internals.h"

#include "../csharp_script.h"
#include "../utils/macros.h"
#include "gd_mono_utils.h"

#include "core/debugger/engine_debugger.h"
#include "core/debugger/script_debugger.h"

#include <mono/metadata/exception.h>

namespace GDMonoInternals {
void unhandled_exception(MonoException *p_exc) {
	mono_print_unhandled_exception((MonoObject *)p_exc);
	gd_unhandled_exception_event(p_exc);

	if (GDMono::get_singleton()->get_unhandled_exception_policy() == GDMono::POLICY_TERMINATE_APP) {
		// Too bad 'mono_invoke_unhandled_exception_hook' is not exposed to embedders
		mono_unhandled_exception((MonoObject *)p_exc);
		GDMono::unhandled_exception_hook((MonoObject *)p_exc, nullptr);
		GD_UNREACHABLE();
	} else {
#ifdef DEBUG_ENABLED
		GDMonoUtils::debug_send_unhandled_exception_error(p_exc);
		if (EngineDebugger::is_active()) {
			EngineDebugger::get_singleton()->poll_events(false);
		}
#endif
	}
}

void gd_unhandled_exception_event(MonoException *p_exc) {
	MonoImage *mono_image = GDMono::get_singleton()->get_core_api_assembly()->get_image();

	MonoClass *gd_klass = mono_class_from_name(mono_image, "Godot", "GD");
	MonoMethod *unhandled_exception_method = mono_class_get_method_from_name(gd_klass, "OnUnhandledException", 1);
	void *args[1];
	args[0] = p_exc;
	mono_runtime_invoke(unhandled_exception_method, nullptr, (void **)args, nullptr);
}
} // namespace GDMonoInternals
