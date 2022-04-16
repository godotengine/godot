/*************************************************************************/
/*  editor_internal_calls.h                                              */
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

#ifndef EDITOR_INTERNAL_CALL_H
#define EDITOR_INTERNAL_CALL_H

#ifdef UNIX_ENABLED
#include <unistd.h> // access
#endif

#include "core/os/os.h"
#include "core/version.h"
#include "editor/debugger/editor_debugger_node.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/plugins/script_editor_plugin.h"
#include "main/main.h"

#include "../csharp_script.h"
#include "../godotsharp_dirs.h"
#include "../utils/osx_utils.h"
#include "code_completion.h"

#include <gdnative/gdnative.h>

#ifdef WIN32
#define GD_CLR_STDCALL __stdcall
#else
#define GD_CLR_STDCALL
#endif

// The order of the fields defined in InternalUnmanagedCallbacks must match the order
// of the defined methods in GodotTools/Internals/Internal.cs
struct InternalUnmanagedCallbacks {
	using Func_godot_icall_GodotSharpDirs_ResMetadataDir = void(GD_CLR_STDCALL *)(godot_string *);
	using Func_godot_icall_GodotSharpDirs_MonoUserDir = void(GD_CLR_STDCALL *)(godot_string *);
	using Func_godot_icall_GodotSharpDirs_BuildLogsDirs = void(GD_CLR_STDCALL *)(godot_string *);
	using Func_godot_icall_GodotSharpDirs_ProjectSlnPath = void(GD_CLR_STDCALL *)(godot_string *);
	using Func_godot_icall_GodotSharpDirs_ProjectCsProjPath = void(GD_CLR_STDCALL *)(godot_string *);
	using Func_godot_icall_GodotSharpDirs_DataEditorToolsDir = void(GD_CLR_STDCALL *)(godot_string *);
	using Func_godot_icall_EditorProgress_Create = void(GD_CLR_STDCALL *)(const godot_string *, const godot_string *, int32_t, bool);
	using Func_godot_icall_EditorProgress_Dispose = void(GD_CLR_STDCALL *)(const godot_string *);
	using Func_godot_icall_EditorProgress_Step = bool(GD_CLR_STDCALL *)(const godot_string *, const godot_string *, int32_t, bool);
	using Func_godot_icall_Internal_FullTemplatesDir = void(GD_CLR_STDCALL *)(godot_string *);
	using Func_godot_icall_Internal_IsOsxAppBundleInstalled = bool(GD_CLR_STDCALL *)(const godot_string *);
	using Func_godot_icall_Internal_GodotIs32Bits = bool(GD_CLR_STDCALL *)();
	using Func_godot_icall_Internal_GodotIsRealTDouble = bool(GD_CLR_STDCALL *)();
	using Func_godot_icall_Internal_GodotMainIteration = void(GD_CLR_STDCALL *)();
	using Func_godot_icall_Internal_IsAssembliesReloadingNeeded = bool(GD_CLR_STDCALL *)();
	using Func_godot_icall_Internal_ReloadAssemblies = void(GD_CLR_STDCALL *)(bool);
	using Func_godot_icall_Internal_EditorDebuggerNodeReloadScripts = void(GD_CLR_STDCALL *)();
	using Func_godot_icall_Internal_ScriptEditorEdit = bool(GD_CLR_STDCALL *)(Resource *, int32_t, int32_t, bool);
	using Func_godot_icall_Internal_EditorNodeShowScriptScreen = void(GD_CLR_STDCALL *)();
	using Func_godot_icall_Internal_EditorRunPlay = void(GD_CLR_STDCALL *)();
	using Func_godot_icall_Internal_EditorRunStop = void(GD_CLR_STDCALL *)();
	using Func_godot_icall_Internal_ScriptEditorDebugger_ReloadScripts = void(GD_CLR_STDCALL *)();
	using Func_godot_icall_Internal_CodeCompletionRequest = void(GD_CLR_STDCALL *)(int32_t, const godot_string *, godot_packed_string_array *);
	using Func_godot_icall_Globals_EditorScale = float(GD_CLR_STDCALL *)();
	using Func_godot_icall_Globals_GlobalDef = void(GD_CLR_STDCALL *)(const godot_string *, const godot_variant *, bool, godot_variant *);
	using Func_godot_icall_Globals_EditorDef = void(GD_CLR_STDCALL *)(const godot_string *, const godot_variant *, bool, godot_variant *);
	using Func_godot_icall_Globals_EditorShortcut = void(GD_CLR_STDCALL *)(const godot_string *, godot_variant *);
	using Func_godot_icall_Globals_TTR = void(GD_CLR_STDCALL *)(const godot_string *, godot_string *);
	using Func_godot_icall_Utils_OS_GetPlatformName = void(GD_CLR_STDCALL *)(godot_string *);
	using Func_godot_icall_Utils_OS_UnixFileHasExecutableAccess = bool(GD_CLR_STDCALL *)(const godot_string *);

	Func_godot_icall_GodotSharpDirs_ResMetadataDir godot_icall_GodotSharpDirs_ResMetadataDir;
	Func_godot_icall_GodotSharpDirs_MonoUserDir godot_icall_GodotSharpDirs_MonoUserDir;
	Func_godot_icall_GodotSharpDirs_BuildLogsDirs godot_icall_GodotSharpDirs_BuildLogsDirs;
	Func_godot_icall_GodotSharpDirs_ProjectSlnPath godot_icall_GodotSharpDirs_ProjectSlnPath;
	Func_godot_icall_GodotSharpDirs_ProjectCsProjPath godot_icall_GodotSharpDirs_ProjectCsProjPath;
	Func_godot_icall_GodotSharpDirs_DataEditorToolsDir godot_icall_GodotSharpDirs_DataEditorToolsDir;
	Func_godot_icall_EditorProgress_Create godot_icall_EditorProgress_Create;
	Func_godot_icall_EditorProgress_Dispose godot_icall_EditorProgress_Dispose;
	Func_godot_icall_EditorProgress_Step godot_icall_EditorProgress_Step;
	Func_godot_icall_Internal_FullTemplatesDir godot_icall_Internal_FullTemplatesDir;
	Func_godot_icall_Internal_IsOsxAppBundleInstalled godot_icall_Internal_IsOsxAppBundleInstalled;
	Func_godot_icall_Internal_GodotIs32Bits godot_icall_Internal_GodotIs32Bits;
	Func_godot_icall_Internal_GodotIsRealTDouble godot_icall_Internal_GodotIsRealTDouble;
	Func_godot_icall_Internal_GodotMainIteration godot_icall_Internal_GodotMainIteration;
	Func_godot_icall_Internal_IsAssembliesReloadingNeeded godot_icall_Internal_IsAssembliesReloadingNeeded;
	Func_godot_icall_Internal_ReloadAssemblies godot_icall_Internal_ReloadAssemblies;
	Func_godot_icall_Internal_EditorDebuggerNodeReloadScripts godot_icall_Internal_EditorDebuggerNodeReloadScripts;
	Func_godot_icall_Internal_ScriptEditorEdit godot_icall_Internal_ScriptEditorEdit;
	Func_godot_icall_Internal_EditorNodeShowScriptScreen godot_icall_Internal_EditorNodeShowScriptScreen;
	Func_godot_icall_Internal_EditorRunPlay godot_icall_Internal_EditorRunPlay;
	Func_godot_icall_Internal_EditorRunStop godot_icall_Internal_EditorRunStop;
	Func_godot_icall_Internal_ScriptEditorDebugger_ReloadScripts godot_icall_Internal_ScriptEditorDebugger_ReloadScripts;
	Func_godot_icall_Internal_CodeCompletionRequest godot_icall_Internal_CodeCompletionRequest;
	Func_godot_icall_Globals_EditorScale godot_icall_Globals_EditorScale;
	Func_godot_icall_Globals_GlobalDef godot_icall_Globals_GlobalDef;
	Func_godot_icall_Globals_EditorDef godot_icall_Globals_EditorDef;
	Func_godot_icall_Globals_EditorShortcut godot_icall_Globals_EditorShortcut;
	Func_godot_icall_Globals_TTR godot_icall_Globals_TTR;
	Func_godot_icall_Utils_OS_GetPlatformName godot_icall_Utils_OS_GetPlatformName;
	Func_godot_icall_Utils_OS_UnixFileHasExecutableAccess godot_icall_Utils_OS_UnixFileHasExecutableAccess;

	static InternalUnmanagedCallbacks create();
};

#endif // EDITOR_INTERNAL_CALL_H
