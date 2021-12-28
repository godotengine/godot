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

#include <stdint.h>

#include "../csharp_script.h"
#include "../mono_gc_handle.h"
#include "core/object/object.h"
#include "core/string/string_name.h"
#include "core/string/ustring.h"
#include "core/variant/callable.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"

class CSharpScript;

namespace GDMonoCache {

#ifdef WIN32
#define GD_CLR_STDCALL __stdcall
#else
#define GD_CLR_STDCALL
#endif

struct ManagedCallbacks {
	using FuncSignalAwaiter_SignalCallback = void(GD_CLR_STDCALL *)(GCHandleIntPtr, const Variant **, int32_t, bool *);
	using FuncDelegateUtils_InvokeWithVariantArgs = void(GD_CLR_STDCALL *)(GCHandleIntPtr, const Variant **, uint32_t, const Variant *);
	using FuncDelegateUtils_DelegateEquals = bool(GD_CLR_STDCALL *)(GCHandleIntPtr, GCHandleIntPtr);
	using FuncScriptManagerBridge_FrameCallback = void(GD_CLR_STDCALL *)();
	using FuncScriptManagerBridge_CreateManagedForGodotObjectBinding = GCHandleIntPtr(GD_CLR_STDCALL *)(const StringName *, Object *);
	using FuncScriptManagerBridge_CreateManagedForGodotObjectScriptInstance = bool(GD_CLR_STDCALL *)(const CSharpScript *, Object *, const Variant **, int);
	using FuncScriptManagerBridge_GetScriptNativeName = void(GD_CLR_STDCALL *)(const CSharpScript *, StringName *);
	using FuncScriptManagerBridge_SetGodotObjectPtr = void(GD_CLR_STDCALL *)(GCHandleIntPtr, Object *);
	using FuncScriptManagerBridge_RaiseEventSignal = void(GD_CLR_STDCALL *)(GCHandleIntPtr, const StringName *, const Variant **, int, bool *);
	using FuncScriptManagerBridge_GetScriptSignalList = void(GD_CLR_STDCALL *)(const CSharpScript *, Dictionary *);
	using FuncScriptManagerBridge_HasScriptSignal = bool(GD_CLR_STDCALL *)(const CSharpScript *, const String *);
	using FuncScriptManagerBridge_HasMethodUnknownParams = bool(GD_CLR_STDCALL *)(const CSharpScript *, const String *, bool);
	using FuncScriptManagerBridge_ScriptIsOrInherits = bool(GD_CLR_STDCALL *)(const CSharpScript *, const CSharpScript *);
	using FuncScriptManagerBridge_AddScriptBridge = bool(GD_CLR_STDCALL *)(const CSharpScript *, const String *);
	using FuncScriptManagerBridge_RemoveScriptBridge = void(GD_CLR_STDCALL *)(const CSharpScript *);
	using FuncScriptManagerBridge_UpdateScriptClassInfo = void(GD_CLR_STDCALL *)(const CSharpScript *, bool *, Dictionary *);
	using FuncScriptManagerBridge_SwapGCHandleForType = bool(GD_CLR_STDCALL *)(GCHandleIntPtr, GCHandleIntPtr *, bool);
	using FuncCSharpInstanceBridge_Call = bool(GD_CLR_STDCALL *)(GCHandleIntPtr, const StringName *, const Variant **, int, Callable::CallError *, Variant *);
	using FuncCSharpInstanceBridge_Set = bool(GD_CLR_STDCALL *)(GCHandleIntPtr, const StringName *, const Variant *);
	using FuncCSharpInstanceBridge_Get = bool(GD_CLR_STDCALL *)(GCHandleIntPtr, const StringName *, Variant *);
	using FuncCSharpInstanceBridge_CallDispose = void(GD_CLR_STDCALL *)(GCHandleIntPtr, bool);
	using FuncCSharpInstanceBridge_CallToString = void(GD_CLR_STDCALL *)(GCHandleIntPtr, String *, bool *);
	using FuncGCHandleBridge_FreeGCHandle = void(GD_CLR_STDCALL *)(GCHandleIntPtr);
	using FuncDebuggingUtils_InstallTraceListener = void(GD_CLR_STDCALL *)();
	using FuncDispatcher_InitializeDefaultGodotTaskScheduler = void(GD_CLR_STDCALL *)();
	using FuncDisposablesTracker_OnGodotShuttingDown = void(GD_CLR_STDCALL *)();

	FuncSignalAwaiter_SignalCallback SignalAwaiter_SignalCallback;
	FuncDelegateUtils_InvokeWithVariantArgs DelegateUtils_InvokeWithVariantArgs;
	FuncDelegateUtils_DelegateEquals DelegateUtils_DelegateEquals;
	FuncScriptManagerBridge_FrameCallback ScriptManagerBridge_FrameCallback;
	FuncScriptManagerBridge_CreateManagedForGodotObjectBinding ScriptManagerBridge_CreateManagedForGodotObjectBinding;
	FuncScriptManagerBridge_CreateManagedForGodotObjectScriptInstance ScriptManagerBridge_CreateManagedForGodotObjectScriptInstance;
	FuncScriptManagerBridge_GetScriptNativeName ScriptManagerBridge_GetScriptNativeName;
	FuncScriptManagerBridge_SetGodotObjectPtr ScriptManagerBridge_SetGodotObjectPtr;
	FuncScriptManagerBridge_RaiseEventSignal ScriptManagerBridge_RaiseEventSignal;
	FuncScriptManagerBridge_GetScriptSignalList ScriptManagerBridge_GetScriptSignalList;
	FuncScriptManagerBridge_HasScriptSignal ScriptManagerBridge_HasScriptSignal;
	FuncScriptManagerBridge_HasMethodUnknownParams ScriptManagerBridge_HasMethodUnknownParams;
	FuncScriptManagerBridge_ScriptIsOrInherits ScriptManagerBridge_ScriptIsOrInherits;
	FuncScriptManagerBridge_AddScriptBridge ScriptManagerBridge_AddScriptBridge;
	FuncScriptManagerBridge_RemoveScriptBridge ScriptManagerBridge_RemoveScriptBridge;
	FuncScriptManagerBridge_UpdateScriptClassInfo ScriptManagerBridge_UpdateScriptClassInfo;
	FuncScriptManagerBridge_SwapGCHandleForType ScriptManagerBridge_SwapGCHandleForType;
	FuncCSharpInstanceBridge_Call CSharpInstanceBridge_Call;
	FuncCSharpInstanceBridge_Set CSharpInstanceBridge_Set;
	FuncCSharpInstanceBridge_Get CSharpInstanceBridge_Get;
	FuncCSharpInstanceBridge_CallDispose CSharpInstanceBridge_CallDispose;
	FuncCSharpInstanceBridge_CallToString CSharpInstanceBridge_CallToString;
	FuncGCHandleBridge_FreeGCHandle GCHandleBridge_FreeGCHandle;
	FuncDebuggingUtils_InstallTraceListener DebuggingUtils_InstallTraceListener;
	FuncDispatcher_InitializeDefaultGodotTaskScheduler Dispatcher_InitializeDefaultGodotTaskScheduler;
	FuncDisposablesTracker_OnGodotShuttingDown DisposablesTracker_OnGodotShuttingDown;
};

extern ManagedCallbacks managed_callbacks;
extern bool godot_api_cache_updated;

void update_godot_api_cache(const ManagedCallbacks &p_managed_callbacks);

inline void clear_godot_api_cache() {
	managed_callbacks = ManagedCallbacks();
}
} // namespace GDMonoCache

#endif // GD_MONO_CACHE_H
