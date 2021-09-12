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

#include "gd_mono_method_thunk.h"

class CSharpScript;

namespace GDMonoCache {

struct CachedData {
	// Mono method thunks require structs to be boxed, even if passed by ref (out, ref, in).
	// As such we need to use pointers instead for now, instead of by ref parameters.

	GDMonoMethodThunk<GCHandleIntPtr, const Variant **, int, bool *> methodthunk_SignalAwaiter_SignalCallback;

	GDMonoMethodThunk<GCHandleIntPtr, const Variant **, uint32_t, const Variant *> methodthunk_DelegateUtils_InvokeWithVariantArgs;
	GDMonoMethodThunkR<bool, GCHandleIntPtr, GCHandleIntPtr> methodthunk_DelegateUtils_DelegateEquals;

	GDMonoMethodThunk<> methodthunk_ScriptManagerBridge_FrameCallback;
	GDMonoMethodThunkR<GCHandleIntPtr, const StringName *, Object *> methodthunk_ScriptManagerBridge_CreateManagedForGodotObjectBinding;
	GDMonoMethodThunk<const CSharpScript *, Object *, const Variant **, int> methodthunk_ScriptManagerBridge_CreateManagedForGodotObjectScriptInstance;
	GDMonoMethodThunk<const CSharpScript *, StringName *> methodthunk_ScriptManagerBridge_GetScriptNativeName;
	GDMonoMethodThunk<MonoReflectionAssembly *> methodthunk_ScriptManagerBridge_LookupScriptsInAssembly;
	GDMonoMethodThunk<GCHandleIntPtr, Object *> methodthunk_ScriptManagerBridge_SetGodotObjectPtr;
	GDMonoMethodThunk<GCHandleIntPtr, const StringName *, const Variant **, int, bool *> methodthunk_ScriptManagerBridge_RaiseEventSignal;
	GDMonoMethodThunk<const CSharpScript *, Dictionary *> methodthunk_ScriptManagerBridge_GetScriptSignalList;
	GDMonoMethodThunkR<bool, const CSharpScript *, const String *> methodthunk_ScriptManagerBridge_HasScriptSignal;
	GDMonoMethodThunkR<bool, const CSharpScript *, const String *, bool> methodthunk_ScriptManagerBridge_HasMethodUnknownParams;
	GDMonoMethodThunkR<bool, const CSharpScript *, const CSharpScript *> methodthunk_ScriptManagerBridge_ScriptIsOrInherits;
	GDMonoMethodThunkR<bool, const CSharpScript *, const String *> methodthunk_ScriptManagerBridge_AddScriptBridge;
	GDMonoMethodThunk<const CSharpScript *> methodthunk_ScriptManagerBridge_RemoveScriptBridge;
	GDMonoMethodThunk<const CSharpScript *, bool *, Dictionary *> methodthunk_ScriptManagerBridge_UpdateScriptClassInfo;
	GDMonoMethodThunkR<bool, GCHandleIntPtr, GCHandleIntPtr *, bool> methodthunk_ScriptManagerBridge_SwapGCHandleForType;

	GDMonoMethodThunk<GCHandleIntPtr, const StringName *, const Variant **, int, Callable::CallError *, Variant *> methodthunk_CSharpInstanceBridge_Call;
	GDMonoMethodThunkR<bool, GCHandleIntPtr, const StringName *, const Variant *> methodthunk_CSharpInstanceBridge_Set;
	GDMonoMethodThunkR<bool, GCHandleIntPtr, const StringName *, Variant *> methodthunk_CSharpInstanceBridge_Get;
	GDMonoMethodThunk<GCHandleIntPtr, bool> methodthunk_CSharpInstanceBridge_CallDispose;
	GDMonoMethodThunk<GCHandleIntPtr, String *, bool *> methodthunk_CSharpInstanceBridge_CallToString;

	GDMonoMethodThunk<GCHandleIntPtr> methodthunk_GCHandleBridge_FreeGCHandle;

	GDMonoMethodThunk<> methodthunk_DebuggingUtils_InstallTraceListener;

	bool godot_api_cache_updated = false;
};

extern CachedData cached_data;

void update_godot_api_cache();

inline void clear_godot_api_cache() {
	cached_data = CachedData();
}
} // namespace GDMonoCache

#endif // GD_MONO_CACHE_H
