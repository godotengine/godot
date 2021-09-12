/*************************************************************************/
/*  gd_mono_cache.cpp                                                    */
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

#include "gd_mono_cache.h"

#include "gd_mono.h"
#include "gd_mono_utils.h"

namespace GDMonoCache {

CachedData cached_data;

static MonoMethod *get_mono_method(MonoClass *p_mono_class, const char *p_method_name, int p_param_count) {
	ERR_FAIL_NULL_V(p_mono_class, nullptr);
	return mono_class_get_method_from_name(p_mono_class, p_method_name, p_param_count);
}

static MonoClass *get_mono_class(GDMonoAssembly *p_assembly, const char *p_namespace, const char *p_name) {
	ERR_FAIL_NULL_V(p_assembly->get_image(), nullptr);
	return mono_class_from_name(p_assembly->get_image(), p_namespace, p_name);
}

void update_godot_api_cache() {
#define CACHE_METHOD_THUNK_AND_CHECK_IMPL(m_var, m_val)                                         \
	{                                                                                           \
		CRASH_COND(!m_var.is_null());                                                           \
		val = m_val;                                                                            \
		ERR_FAIL_COND_MSG(val == nullptr, "Mono Cache: Method for member " #m_var " is null."); \
		m_var.set_from_method(val);                                                             \
		ERR_FAIL_COND_MSG(m_var.is_null(), "Mono Cache: Member " #m_var " is null.");           \
	}

#define CACHE_METHOD_THUNK_AND_CHECK(m_class, m_method, m_val) CACHE_METHOD_THUNK_AND_CHECK_IMPL(cached_data.methodthunk_##m_class##_##m_method, m_val)

#define GODOT_API_CLASS(m_class) (get_mono_class(GDMono::get_singleton()->get_core_api_assembly(), BINDINGS_NAMESPACE, #m_class))
#define GODOT_API_BRIDGE_CLASS(m_class) (get_mono_class(GDMono::get_singleton()->get_core_api_assembly(), BINDINGS_NAMESPACE_BRIDGE, #m_class))

	MonoMethod *val = nullptr;

	CACHE_METHOD_THUNK_AND_CHECK(SignalAwaiter, SignalCallback, get_mono_method(GODOT_API_CLASS(SignalAwaiter), "SignalCallback", 4));

	CACHE_METHOD_THUNK_AND_CHECK(DelegateUtils, InvokeWithVariantArgs, get_mono_method(GODOT_API_CLASS(DelegateUtils), "InvokeWithVariantArgs", 4));
	CACHE_METHOD_THUNK_AND_CHECK(DelegateUtils, DelegateEquals, get_mono_method(GODOT_API_CLASS(DelegateUtils), "DelegateEquals", 2));

	CACHE_METHOD_THUNK_AND_CHECK(ScriptManagerBridge, FrameCallback, get_mono_method(GODOT_API_BRIDGE_CLASS(ScriptManagerBridge), "FrameCallback", 0));
	CACHE_METHOD_THUNK_AND_CHECK(ScriptManagerBridge, CreateManagedForGodotObjectBinding, get_mono_method(GODOT_API_BRIDGE_CLASS(ScriptManagerBridge), "CreateManagedForGodotObjectBinding", 2));
	CACHE_METHOD_THUNK_AND_CHECK(ScriptManagerBridge, CreateManagedForGodotObjectScriptInstance, get_mono_method(GODOT_API_BRIDGE_CLASS(ScriptManagerBridge), "CreateManagedForGodotObjectScriptInstance", 4));
	CACHE_METHOD_THUNK_AND_CHECK(ScriptManagerBridge, GetScriptNativeName, get_mono_method(GODOT_API_BRIDGE_CLASS(ScriptManagerBridge), "GetScriptNativeName", 2));
	CACHE_METHOD_THUNK_AND_CHECK(ScriptManagerBridge, LookupScriptsInAssembly, get_mono_method(GODOT_API_BRIDGE_CLASS(ScriptManagerBridge), "LookupScriptsInAssembly", 1));
	CACHE_METHOD_THUNK_AND_CHECK(ScriptManagerBridge, SetGodotObjectPtr, get_mono_method(GODOT_API_BRIDGE_CLASS(ScriptManagerBridge), "SetGodotObjectPtr", 2));
	CACHE_METHOD_THUNK_AND_CHECK(ScriptManagerBridge, RaiseEventSignal, get_mono_method(GODOT_API_BRIDGE_CLASS(ScriptManagerBridge), "RaiseEventSignal", 5));
	CACHE_METHOD_THUNK_AND_CHECK(ScriptManagerBridge, GetScriptSignalList, get_mono_method(GODOT_API_BRIDGE_CLASS(ScriptManagerBridge), "GetScriptSignalList", 2));
	CACHE_METHOD_THUNK_AND_CHECK(ScriptManagerBridge, HasScriptSignal, get_mono_method(GODOT_API_BRIDGE_CLASS(ScriptManagerBridge), "HasScriptSignal", 2));
	CACHE_METHOD_THUNK_AND_CHECK(ScriptManagerBridge, HasMethodUnknownParams, get_mono_method(GODOT_API_BRIDGE_CLASS(ScriptManagerBridge), "HasMethodUnknownParams", 3));
	CACHE_METHOD_THUNK_AND_CHECK(ScriptManagerBridge, ScriptIsOrInherits, get_mono_method(GODOT_API_BRIDGE_CLASS(ScriptManagerBridge), "ScriptIsOrInherits", 2));
	CACHE_METHOD_THUNK_AND_CHECK(ScriptManagerBridge, AddScriptBridge, get_mono_method(GODOT_API_BRIDGE_CLASS(ScriptManagerBridge), "AddScriptBridge", 2));
	CACHE_METHOD_THUNK_AND_CHECK(ScriptManagerBridge, RemoveScriptBridge, get_mono_method(GODOT_API_BRIDGE_CLASS(ScriptManagerBridge), "RemoveScriptBridge", 1));
	CACHE_METHOD_THUNK_AND_CHECK(ScriptManagerBridge, UpdateScriptClassInfo, get_mono_method(GODOT_API_BRIDGE_CLASS(ScriptManagerBridge), "UpdateScriptClassInfo", 3));
	CACHE_METHOD_THUNK_AND_CHECK(ScriptManagerBridge, SwapGCHandleForType, get_mono_method(GODOT_API_BRIDGE_CLASS(ScriptManagerBridge), "SwapGCHandleForType", 3));

	CACHE_METHOD_THUNK_AND_CHECK(CSharpInstanceBridge, Call, get_mono_method(GODOT_API_BRIDGE_CLASS(CSharpInstanceBridge), "Call", 6));
	CACHE_METHOD_THUNK_AND_CHECK(CSharpInstanceBridge, Set, get_mono_method(GODOT_API_BRIDGE_CLASS(CSharpInstanceBridge), "Set", 3));
	CACHE_METHOD_THUNK_AND_CHECK(CSharpInstanceBridge, Get, get_mono_method(GODOT_API_BRIDGE_CLASS(CSharpInstanceBridge), "Get", 3));
	CACHE_METHOD_THUNK_AND_CHECK(CSharpInstanceBridge, CallDispose, get_mono_method(GODOT_API_BRIDGE_CLASS(CSharpInstanceBridge), "CallDispose", 2));
	CACHE_METHOD_THUNK_AND_CHECK(CSharpInstanceBridge, CallToString, get_mono_method(GODOT_API_BRIDGE_CLASS(CSharpInstanceBridge), "CallToString", 3));

	CACHE_METHOD_THUNK_AND_CHECK(GCHandleBridge, FreeGCHandle, get_mono_method(GODOT_API_BRIDGE_CLASS(GCHandleBridge), "FreeGCHandle", 1));

	CACHE_METHOD_THUNK_AND_CHECK(DebuggingUtils, InstallTraceListener, get_mono_method(GODOT_API_CLASS(DebuggingUtils), "InstallTraceListener", 0));

	MonoException *exc = nullptr;
	MonoMethod *init_default_godot_task_scheduler =
			get_mono_method(GODOT_API_CLASS(Dispatcher), "InitializeDefaultGodotTaskScheduler", 0);
	ERR_FAIL_COND_MSG(init_default_godot_task_scheduler == nullptr,
			"Mono Cache: InitializeDefaultGodotTaskScheduler is null.");
	mono_runtime_invoke(init_default_godot_task_scheduler, nullptr, nullptr, (MonoObject **)&exc);

	if (exc) {
		GDMonoUtils::debug_unhandled_exception(exc);
	}

	cached_data.godot_api_cache_updated = true;
}
} // namespace GDMonoCache
