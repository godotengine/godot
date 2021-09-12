/*************************************************************************/
/*  editor_internal_calls.cpp                                            */
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

#include "editor_internal_calls.h"

#ifdef UNIX_ENABLED
#include <unistd.h> // access
#endif

#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "core/version.h"
#include "editor/debugger/editor_debugger_node.h"
#include "editor/editor_node.h"
#include "editor/editor_paths.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/plugins/script_editor_plugin.h"
#include "main/main.h"

#include "../csharp_script.h"
#include "../godotsharp_dirs.h"
#include "../utils/macos_utils.h"
#include "code_completion.h"
#include "godotsharp_export.h"

#include "../interop_types.h"

void godot_icall_GodotSharpDirs_ResMetadataDir(godot_string *r_dest) {
	memnew_placement(r_dest, String(GodotSharpDirs::get_res_metadata_dir()));
}

void godot_icall_GodotSharpDirs_ResTempAssembliesBaseDir(godot_string *r_dest) {
	memnew_placement(r_dest, String(GodotSharpDirs::get_res_temp_assemblies_base_dir()));
}

void godot_icall_GodotSharpDirs_MonoUserDir(godot_string *r_dest) {
	memnew_placement(r_dest, String(GodotSharpDirs::get_mono_user_dir()));
}

void godot_icall_GodotSharpDirs_BuildLogsDirs(godot_string *r_dest) {
#ifdef TOOLS_ENABLED
	memnew_placement(r_dest, String(GodotSharpDirs::get_build_logs_dir()));
#else
	return nullptr;
#endif
}

void godot_icall_GodotSharpDirs_ProjectSlnPath(godot_string *r_dest) {
#ifdef TOOLS_ENABLED
	memnew_placement(r_dest, String(GodotSharpDirs::get_project_sln_path()));
#else
	return nullptr;
#endif
}

void godot_icall_GodotSharpDirs_ProjectCsProjPath(godot_string *r_dest) {
#ifdef TOOLS_ENABLED
	memnew_placement(r_dest, String(GodotSharpDirs::get_project_csproj_path()));
#else
	return nullptr;
#endif
}

void godot_icall_GodotSharpDirs_DataEditorToolsDir(godot_string *r_dest) {
#ifdef TOOLS_ENABLED
	memnew_placement(r_dest, String(GodotSharpDirs::get_data_editor_tools_dir()));
#else
	return nullptr;
#endif
}

void godot_icall_EditorProgress_Create(const godot_string *p_task, const godot_string *p_label, int32_t p_amount, bool p_can_cancel) {
	String task = *reinterpret_cast<const String *>(p_task);
	String label = *reinterpret_cast<const String *>(p_label);
	EditorNode::progress_add_task(task, label, p_amount, (bool)p_can_cancel);
}

void godot_icall_EditorProgress_Dispose(const godot_string *p_task) {
	String task = *reinterpret_cast<const String *>(p_task);
	EditorNode::progress_end_task(task);
}

bool godot_icall_EditorProgress_Step(const godot_string *p_task, const godot_string *p_state, int32_t p_step, bool p_force_refresh) {
	String task = *reinterpret_cast<const String *>(p_task);
	String state = *reinterpret_cast<const String *>(p_state);
	return EditorNode::progress_task_step(task, state, p_step, (bool)p_force_refresh);
}

uint32_t godot_icall_ExportPlugin_GetExportedAssemblyDependencies(const godot_dictionary *p_initial_assemblies,
		const godot_string *p_build_config, const godot_string *p_custom_bcl_dir, godot_dictionary *r_assembly_dependencies) {
	Dictionary initial_dependencies = *reinterpret_cast<const Dictionary *>(p_initial_assemblies);
	String build_config = *reinterpret_cast<const String *>(p_build_config);
	String custom_bcl_dir = *reinterpret_cast<const String *>(p_custom_bcl_dir);
	Dictionary assembly_dependencies = *reinterpret_cast<Dictionary *>(r_assembly_dependencies);

	return GodotSharpExport::get_exported_assembly_dependencies(initial_dependencies, build_config, custom_bcl_dir, assembly_dependencies);
}

void godot_icall_Internal_FullExportTemplatesDir(godot_string *r_dest) {
	String full_templates_dir = EditorPaths::get_singleton()->get_export_templates_dir().plus_file(VERSION_FULL_CONFIG);
	memnew_placement(r_dest, String(full_templates_dir));
}

bool godot_icall_Internal_IsMacOSAppBundleInstalled(const godot_string *p_bundle_id) {
#ifdef MACOS_ENABLED
	String bundle_id = *reinterpret_cast<const String *>(p_bundle_id);
	return (bool)macos_is_app_bundle_installed(bundle_id);
#else
	(void)p_bundle_id; // UNUSED
	return (bool)false;
#endif
}

bool godot_icall_Internal_GodotIs32Bits() {
	return sizeof(void *) == 4;
}

bool godot_icall_Internal_GodotIsRealTDouble() {
#ifdef REAL_T_IS_DOUBLE
	return (bool)true;
#else
	return (bool)false;
#endif
}

void godot_icall_Internal_GodotMainIteration() {
	Main::iteration();
}

bool godot_icall_Internal_IsAssembliesReloadingNeeded() {
#ifdef GD_MONO_HOT_RELOAD
	return (bool)CSharpLanguage::get_singleton()->is_assembly_reloading_needed();
#else
	return (bool)false;
#endif
}

void godot_icall_Internal_ReloadAssemblies(bool p_soft_reload) {
#ifdef GD_MONO_HOT_RELOAD
	mono_bind::GodotSharp::get_singleton()->call_deferred(SNAME("_reload_assemblies"), (bool)p_soft_reload);
#endif
}

void godot_icall_Internal_EditorDebuggerNodeReloadScripts() {
	EditorDebuggerNode::get_singleton()->reload_scripts();
}

bool godot_icall_Internal_ScriptEditorEdit(Resource *p_resource, int32_t p_line, int32_t p_col, bool p_grab_focus) {
	Ref<Resource> resource = p_resource;
	return (bool)ScriptEditor::get_singleton()->edit(resource, p_line, p_col, (bool)p_grab_focus);
}

void godot_icall_Internal_EditorNodeShowScriptScreen() {
	EditorNode::get_singleton()->call("_editor_select", EditorNode::EDITOR_SCRIPT);
}

void godot_icall_Internal_MonoWindowsInstallRoot(godot_string *r_dest) {
#ifdef WINDOWS_ENABLED
	String install_root_dir = GDMono::get_singleton()->get_mono_reg_info().install_root_dir;
	memnew_placement(r_dest, String(install_root_dir));
#else
	memnew_placement(r_dest, String);
	return;
#endif
}

void godot_icall_Internal_EditorRunPlay() {
	EditorNode::get_singleton()->run_play();
}

void godot_icall_Internal_EditorRunStop() {
	EditorNode::get_singleton()->run_stop();
}

void godot_icall_Internal_ScriptEditorDebugger_ReloadScripts() {
	EditorDebuggerNode *ed = EditorDebuggerNode::get_singleton();
	if (ed) {
		ed->reload_scripts();
	}
}

void godot_icall_Internal_CodeCompletionRequest(int32_t p_kind, const godot_string *p_script_file, godot_packed_array *r_ret) {
	String script_file = *reinterpret_cast<const String *>(p_script_file);
	PackedStringArray suggestions = gdmono::get_code_completion((gdmono::CompletionKind)p_kind, script_file);
	memnew_placement(r_ret, PackedStringArray(suggestions));
}

float godot_icall_Globals_EditorScale() {
	return EDSCALE;
}

void godot_icall_Globals_GlobalDef(const godot_string *p_setting, const godot_variant *p_default_value, bool p_restart_if_changed, godot_variant *r_result) {
	String setting = *reinterpret_cast<const String *>(p_setting);
	Variant default_value = *reinterpret_cast<const Variant *>(p_default_value);
	Variant result = _GLOBAL_DEF(setting, default_value, (bool)p_restart_if_changed);
	memnew_placement(r_result, Variant(result));
}

void godot_icall_Globals_EditorDef(const godot_string *p_setting, const godot_variant *p_default_value, bool p_restart_if_changed, godot_variant *r_result) {
	String setting = *reinterpret_cast<const String *>(p_setting);
	Variant default_value = *reinterpret_cast<const Variant *>(p_default_value);
	Variant result = _EDITOR_DEF(setting, default_value, (bool)p_restart_if_changed);
	memnew_placement(r_result, Variant(result));
}

void godot_icall_Globals_EditorShortcut(const godot_string *p_setting, godot_variant *r_result) {
	String setting = *reinterpret_cast<const String *>(p_setting);
	Ref<Shortcut> result = ED_GET_SHORTCUT(setting);
	memnew_placement(r_result, Variant(result));
}

void godot_icall_Globals_TTR(const godot_string *p_text, godot_string *r_dest) {
	String text = *reinterpret_cast<const String *>(p_text);
	memnew_placement(r_dest, String(TTR(text)));
}

void godot_icall_Utils_OS_GetPlatformName(godot_string *r_dest) {
	String os_name = OS::get_singleton()->get_name();
	memnew_placement(r_dest, String(os_name));
}

bool godot_icall_Utils_OS_UnixFileHasExecutableAccess(const godot_string *p_file_path) {
#ifdef UNIX_ENABLED
	String file_path = *reinterpret_cast<const String *>(p_file_path);
	return access(file_path.utf8().get_data(), X_OK) == 0;
#else
	ERR_FAIL_V(false);
#endif
}

void register_editor_internal_calls() {
	// GodotSharpDirs
	GDMonoUtils::add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_ResMetadataDir", godot_icall_GodotSharpDirs_ResMetadataDir);
	GDMonoUtils::add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_ResTempAssembliesBaseDir", godot_icall_GodotSharpDirs_ResTempAssembliesBaseDir);
	GDMonoUtils::add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_MonoUserDir", godot_icall_GodotSharpDirs_MonoUserDir);
	GDMonoUtils::add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_BuildLogsDirs", godot_icall_GodotSharpDirs_BuildLogsDirs);
	GDMonoUtils::add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_ProjectSlnPath", godot_icall_GodotSharpDirs_ProjectSlnPath);
	GDMonoUtils::add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_ProjectCsProjPath", godot_icall_GodotSharpDirs_ProjectCsProjPath);
	GDMonoUtils::add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_DataEditorToolsDir", godot_icall_GodotSharpDirs_DataEditorToolsDir);

	// EditorProgress
	GDMonoUtils::add_internal_call("GodotTools.Internals.EditorProgress::internal_Create", godot_icall_EditorProgress_Create);
	GDMonoUtils::add_internal_call("GodotTools.Internals.EditorProgress::internal_Dispose", godot_icall_EditorProgress_Dispose);
	GDMonoUtils::add_internal_call("GodotTools.Internals.EditorProgress::internal_Step", godot_icall_EditorProgress_Step);

	// ExportPlugin
	GDMonoUtils::add_internal_call("GodotTools.Export.ExportPlugin::internal_GetExportedAssemblyDependencies", godot_icall_ExportPlugin_GetExportedAssemblyDependencies);

	// Internals
	GDMonoUtils::add_internal_call("GodotTools.Internals.Internal::internal_FullExportTemplatesDir", godot_icall_Internal_FullExportTemplatesDir);
	GDMonoUtils::add_internal_call("GodotTools.Internals.Internal::internal_IsMacOSAppBundleInstalled", godot_icall_Internal_IsMacOSAppBundleInstalled);
	GDMonoUtils::add_internal_call("GodotTools.Internals.Internal::internal_GodotIs32Bits", godot_icall_Internal_GodotIs32Bits);
	GDMonoUtils::add_internal_call("GodotTools.Internals.Internal::internal_GodotIsRealTDouble", godot_icall_Internal_GodotIsRealTDouble);
	GDMonoUtils::add_internal_call("GodotTools.Internals.Internal::internal_GodotMainIteration", godot_icall_Internal_GodotMainIteration);
	GDMonoUtils::add_internal_call("GodotTools.Internals.Internal::internal_IsAssembliesReloadingNeeded", godot_icall_Internal_IsAssembliesReloadingNeeded);
	GDMonoUtils::add_internal_call("GodotTools.Internals.Internal::internal_ReloadAssemblies", godot_icall_Internal_ReloadAssemblies);
	GDMonoUtils::add_internal_call("GodotTools.Internals.Internal::internal_EditorDebuggerNodeReloadScripts", godot_icall_Internal_EditorDebuggerNodeReloadScripts);
	GDMonoUtils::add_internal_call("GodotTools.Internals.Internal::internal_ScriptEditorEdit", godot_icall_Internal_ScriptEditorEdit);
	GDMonoUtils::add_internal_call("GodotTools.Internals.Internal::internal_EditorNodeShowScriptScreen", godot_icall_Internal_EditorNodeShowScriptScreen);
	GDMonoUtils::add_internal_call("GodotTools.Internals.Internal::internal_MonoWindowsInstallRoot", godot_icall_Internal_MonoWindowsInstallRoot);
	GDMonoUtils::add_internal_call("GodotTools.Internals.Internal::internal_EditorRunPlay", godot_icall_Internal_EditorRunPlay);
	GDMonoUtils::add_internal_call("GodotTools.Internals.Internal::internal_EditorRunStop", godot_icall_Internal_EditorRunStop);
	GDMonoUtils::add_internal_call("GodotTools.Internals.Internal::internal_ScriptEditorDebugger_ReloadScripts", godot_icall_Internal_ScriptEditorDebugger_ReloadScripts);
	GDMonoUtils::add_internal_call("GodotTools.Internals.Internal::internal_CodeCompletionRequest", godot_icall_Internal_CodeCompletionRequest);

	// Globals
	GDMonoUtils::add_internal_call("GodotTools.Internals.Globals::internal_EditorScale", godot_icall_Globals_EditorScale);
	GDMonoUtils::add_internal_call("GodotTools.Internals.Globals::internal_GlobalDef", godot_icall_Globals_GlobalDef);
	GDMonoUtils::add_internal_call("GodotTools.Internals.Globals::internal_EditorDef", godot_icall_Globals_EditorDef);
	GDMonoUtils::add_internal_call("GodotTools.Internals.Globals::internal_EditorShortcut", godot_icall_Globals_EditorShortcut);
	GDMonoUtils::add_internal_call("GodotTools.Internals.Globals::internal_TTR", godot_icall_Globals_TTR);

	// Utils.OS
	GDMonoUtils::add_internal_call("GodotTools.Utils.OS::GetPlatformName", godot_icall_Utils_OS_GetPlatformName);
	GDMonoUtils::add_internal_call("GodotTools.Utils.OS::UnixFileHasExecutableAccess", godot_icall_Utils_OS_UnixFileHasExecutableAccess);
}
