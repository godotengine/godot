/*************************************************************************/
/*  gd_mono_utils.cpp                                                    */
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

#include "gd_mono_utils.h"

#include <mono/metadata/debug-helpers.h>
#include <mono/metadata/exception.h>

#include "core/debugger/engine_debugger.h"
#include "core/debugger/script_debugger.h"
#include "core/io/dir_access.h"
#include "core/object/ref_counted.h"
#include "core/os/mutex.h"
#include "core/os/os.h"

#ifdef TOOLS_ENABLED
#include "editor/debugger/editor_debugger_node.h"
#endif

#include "../csharp_script.h"
#include "../utils/macros.h"
#include "gd_mono.h"
#include "gd_mono_cache.h"

namespace GDMonoUtils {

void set_main_thread(MonoThread *p_thread) {
	mono_thread_set_main(p_thread);
}

MonoThread *attach_current_thread() {
	ERR_FAIL_COND_V(!GDMono::get_singleton()->is_runtime_initialized(), nullptr);
	MonoDomain *scripts_domain = GDMono::get_singleton()->get_scripts_domain();
#ifndef GD_MONO_SINGLE_APPDOMAIN
	MonoThread *mono_thread = mono_thread_attach(scripts_domain ? scripts_domain : mono_get_root_domain());
#else
	// The scripts domain is the root domain
	MonoThread *mono_thread = mono_thread_attach(scripts_domain);
#endif
	ERR_FAIL_NULL_V(mono_thread, nullptr);
	return mono_thread;
}

void detach_current_thread() {
	ERR_FAIL_COND(!GDMono::get_singleton()->is_runtime_initialized());
	MonoThread *mono_thread = mono_thread_current();
	ERR_FAIL_NULL(mono_thread);
	mono_thread_detach(mono_thread);
}

void detach_current_thread(MonoThread *p_mono_thread) {
	ERR_FAIL_COND(!GDMono::get_singleton()->is_runtime_initialized());
	ERR_FAIL_NULL(p_mono_thread);
	mono_thread_detach(p_mono_thread);
}

MonoThread *get_current_thread() {
	return mono_thread_current();
}

bool is_thread_attached() {
	return mono_domain_get() != nullptr;
}

MonoDomain *create_domain(const String &p_friendly_name) {
	print_verbose("Mono: Creating domain '" + p_friendly_name + "'...");

	MonoDomain *domain = mono_domain_create_appdomain((char *)p_friendly_name.utf8().get_data(), nullptr);

	if (domain) {
		// Workaround to avoid this exception:
		// System.Configuration.ConfigurationErrorsException: Error Initializing the configuration system.
		// ---> System.ArgumentException: The 'ExeConfigFilename' argument cannot be null.
		mono_domain_set_config(domain, ".", "");
	}

	return domain;
}

// TODO:
//  Implement all of the disabled exception logging below. Once we move to .NET 6.
//  It will have to be done from C# as UnmanagedCallersOnly doesn't allow throwing.

#warning TODO
#if 0
String get_exception_name_and_message(MonoException *p_exc) {
	String res;

	MonoClass *klass = mono_object_get_class((MonoObject *)p_exc);
	MonoType *type = mono_class_get_type(klass);

	char *full_name = mono_type_full_name(type);
	res += full_name;
	mono_free(full_name);

	res += ": ";

	MonoProperty *prop = mono_class_get_property_from_name(klass, "Message");
	MonoString *msg = (MonoString *)property_get_value(prop, (MonoObject *)p_exc, nullptr, nullptr);
	res += GDMonoMarshal::mono_string_to_godot(msg);

	return res;
}
#endif

void debug_print_unhandled_exception(MonoException *p_exc) {
	print_unhandled_exception(p_exc);
	debug_send_unhandled_exception_error(p_exc);
}

void debug_send_unhandled_exception_error(MonoException *p_exc) {
#ifdef DEBUG_ENABLED
	if (!EngineDebugger::is_active()) {
#ifdef TOOLS_ENABLED
		if (Engine::get_singleton()->is_editor_hint()) {
#warning TODO
#if 0
			ERR_PRINT(GDMonoUtils::get_exception_name_and_message(p_exc));
#endif
		}
#endif
		return;
	}

	static thread_local bool _recursion_flag_ = false;
	if (_recursion_flag_) {
		return;
	}
	_recursion_flag_ = true;
	SCOPE_EXIT { _recursion_flag_ = false; };

	ScriptLanguage::StackInfo separator;
	separator.file = String();
	separator.func = "--- " + RTR("End of inner exception stack trace") + " ---";
	separator.line = 0;

	Vector<ScriptLanguage::StackInfo> si;
	String exc_msg;

#warning TODO
#if 0
	while (p_exc != nullptr) {
		GDMonoClass *st_klass = CACHED_CLASS(System_Diagnostics_StackTrace);
		MonoObject *stack_trace = mono_object_new(mono_domain_get(), st_klass->get_mono_ptr());

		MonoBoolean need_file_info = true;
		void *ctor_args[2] = { p_exc, &need_file_info };

		MonoException *unexpected_exc = nullptr;
		CACHED_METHOD(System_Diagnostics_StackTrace, ctor_Exception_bool)->invoke_raw(stack_trace, ctor_args, &unexpected_exc);

		if (unexpected_exc) {
			GDMonoInternals::unhandled_exception(unexpected_exc);
			return;
		}

		Vector<ScriptLanguage::StackInfo> _si;
		if (stack_trace != nullptr) {
			_si = CSharpLanguage::get_singleton()->stack_trace_get_info(stack_trace);
			for (int i = _si.size() - 1; i >= 0; i--) {
				si.insert(0, _si[i]);
			}
		}

		exc_msg += (exc_msg.length() > 0 ? " ---> " : "") + GDMonoUtils::get_exception_name_and_message(p_exc);

		GDMonoClass *exc_class = GDMono::get_singleton()->get_class(mono_get_exception_class());
		GDMonoProperty *inner_exc_prop = exc_class->get_property("InnerException");
		CRASH_COND(inner_exc_prop == nullptr);

		MonoObject *inner_exc = inner_exc_prop->get_value((MonoObject *)p_exc);
		if (inner_exc != nullptr) {
			si.insert(0, separator);
		}

		p_exc = (MonoException *)inner_exc;
	}
#endif

	String file = si.size() ? si[0].file : __FILE__;
	String func = si.size() ? si[0].func : FUNCTION_STR;
	int line = si.size() ? si[0].line : __LINE__;
	String error_msg = "Unhandled exception";

	EngineDebugger::get_script_debugger()->send_error(func, file, line, error_msg, exc_msg, true, ERR_HANDLER_ERROR, si);
#endif
}

void debug_unhandled_exception(MonoException *p_exc) {
	GDMonoInternals::unhandled_exception(p_exc); // prints the exception as well
}

void print_unhandled_exception(MonoException *p_exc) {
	mono_print_unhandled_exception((MonoObject *)p_exc);
}

void set_pending_exception(MonoException *p_exc) {
#ifdef NO_PENDING_EXCEPTIONS
	debug_unhandled_exception(p_exc);
#else
	if (get_runtime_invoke_count() == 0) {
		debug_unhandled_exception(p_exc);
		return;
	}

	if (!mono_runtime_set_pending_exception(p_exc, false)) {
		ERR_PRINT("Exception thrown from managed code, but it could not be set as pending:");
		GDMonoUtils::debug_print_unhandled_exception(p_exc);
	}
#endif
}

thread_local int current_invoke_count = 0;

ScopeThreadAttach::ScopeThreadAttach() {
	if (likely(GDMono::get_singleton()->is_runtime_initialized()) && unlikely(!mono_domain_get())) {
		mono_thread = GDMonoUtils::attach_current_thread();
	}
}

ScopeThreadAttach::~ScopeThreadAttach() {
	if (unlikely(mono_thread)) {
		GDMonoUtils::detach_current_thread(mono_thread);
	}
}
} // namespace GDMonoUtils
