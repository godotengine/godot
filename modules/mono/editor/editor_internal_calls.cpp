/*************************************************************************/
/*  editor_internal_calls.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "core/message_queue.h"
#include "core/os/os.h"
#include "core/version.h"
#include "editor/editor_node.h"
#include "editor/plugins/script_editor_plugin.h"
#include "editor/script_editor_debugger.h"
#include "main/main.h"

#include "../csharp_script.h"
#include "../glue/cs_glue_version.gen.h"
#include "../godotsharp_dirs.h"
#include "../mono_gd/gd_mono_marshal.h"
#include "../utils/osx_utils.h"
#include "bindings_generator.h"
#include "godotsharp_export.h"
#include "script_class_parser.h"

MonoString *godot_icall_GodotSharpDirs_ResDataDir() {
	return GDMonoMarshal::mono_string_from_godot(GodotSharpDirs::get_res_data_dir());
}

MonoString *godot_icall_GodotSharpDirs_ResMetadataDir() {
	return GDMonoMarshal::mono_string_from_godot(GodotSharpDirs::get_res_metadata_dir());
}

MonoString *godot_icall_GodotSharpDirs_ResAssembliesBaseDir() {
	return GDMonoMarshal::mono_string_from_godot(GodotSharpDirs::get_res_assemblies_base_dir());
}

MonoString *godot_icall_GodotSharpDirs_ResAssembliesDir() {
	return GDMonoMarshal::mono_string_from_godot(GodotSharpDirs::get_res_assemblies_dir());
}

MonoString *godot_icall_GodotSharpDirs_ResConfigDir() {
	return GDMonoMarshal::mono_string_from_godot(GodotSharpDirs::get_res_config_dir());
}

MonoString *godot_icall_GodotSharpDirs_ResTempDir() {
	return GDMonoMarshal::mono_string_from_godot(GodotSharpDirs::get_res_temp_dir());
}

MonoString *godot_icall_GodotSharpDirs_ResTempAssembliesBaseDir() {
	return GDMonoMarshal::mono_string_from_godot(GodotSharpDirs::get_res_temp_assemblies_base_dir());
}

MonoString *godot_icall_GodotSharpDirs_ResTempAssembliesDir() {
	return GDMonoMarshal::mono_string_from_godot(GodotSharpDirs::get_res_temp_assemblies_dir());
}

MonoString *godot_icall_GodotSharpDirs_MonoUserDir() {
	return GDMonoMarshal::mono_string_from_godot(GodotSharpDirs::get_mono_user_dir());
}

MonoString *godot_icall_GodotSharpDirs_MonoLogsDir() {
	return GDMonoMarshal::mono_string_from_godot(GodotSharpDirs::get_mono_logs_dir());
}

MonoString *godot_icall_GodotSharpDirs_MonoSolutionsDir() {
#ifdef TOOLS_ENABLED
	return GDMonoMarshal::mono_string_from_godot(GodotSharpDirs::get_mono_solutions_dir());
#else
	return NULL;
#endif
}

MonoString *godot_icall_GodotSharpDirs_BuildLogsDirs() {
#ifdef TOOLS_ENABLED
	return GDMonoMarshal::mono_string_from_godot(GodotSharpDirs::get_build_logs_dir());
#else
	return NULL;
#endif
}

MonoString *godot_icall_GodotSharpDirs_ProjectSlnPath() {
#ifdef TOOLS_ENABLED
	return GDMonoMarshal::mono_string_from_godot(GodotSharpDirs::get_project_sln_path());
#else
	return NULL;
#endif
}

MonoString *godot_icall_GodotSharpDirs_ProjectCsProjPath() {
#ifdef TOOLS_ENABLED
	return GDMonoMarshal::mono_string_from_godot(GodotSharpDirs::get_project_csproj_path());
#else
	return NULL;
#endif
}

MonoString *godot_icall_GodotSharpDirs_DataEditorToolsDir() {
#ifdef TOOLS_ENABLED
	return GDMonoMarshal::mono_string_from_godot(GodotSharpDirs::get_data_editor_tools_dir());
#else
	return NULL;
#endif
}

MonoString *godot_icall_GodotSharpDirs_DataEditorPrebuiltApiDir() {
#ifdef TOOLS_ENABLED
	return GDMonoMarshal::mono_string_from_godot(GodotSharpDirs::get_data_editor_prebuilt_api_dir());
#else
	return NULL;
#endif
}

MonoString *godot_icall_GodotSharpDirs_DataMonoEtcDir() {
	return GDMonoMarshal::mono_string_from_godot(GodotSharpDirs::get_data_mono_etc_dir());
}

MonoString *godot_icall_GodotSharpDirs_DataMonoLibDir() {
	return GDMonoMarshal::mono_string_from_godot(GodotSharpDirs::get_data_mono_lib_dir());
}

MonoString *godot_icall_GodotSharpDirs_DataMonoBinDir() {
#ifdef WINDOWS_ENABLED
	return GDMonoMarshal::mono_string_from_godot(GodotSharpDirs::get_data_mono_bin_dir());
#else
	return NULL;
#endif
}

void godot_icall_EditorProgress_Create(MonoString *p_task, MonoString *p_label, int32_t p_amount, MonoBoolean p_can_cancel) {
	String task = GDMonoMarshal::mono_string_to_godot(p_task);
	String label = GDMonoMarshal::mono_string_to_godot(p_label);
	EditorNode::progress_add_task(task, label, p_amount, (bool)p_can_cancel);
}

void godot_icall_EditorProgress_Dispose(MonoString *p_task) {
	String task = GDMonoMarshal::mono_string_to_godot(p_task);
	EditorNode::progress_end_task(task);
}

MonoBoolean godot_icall_EditorProgress_Step(MonoString *p_task, MonoString *p_state, int32_t p_step, MonoBoolean p_force_refresh) {
	String task = GDMonoMarshal::mono_string_to_godot(p_task);
	String state = GDMonoMarshal::mono_string_to_godot(p_state);
	return EditorNode::progress_task_step(task, state, p_step, (bool)p_force_refresh);
}

BindingsGenerator *godot_icall_BindingsGenerator_Ctor() {
	return memnew(BindingsGenerator);
}

void godot_icall_BindingsGenerator_Dtor(BindingsGenerator *p_handle) {
	memdelete(p_handle);
}

MonoBoolean godot_icall_BindingsGenerator_LogPrintEnabled(BindingsGenerator *p_handle) {
	return p_handle->is_log_print_enabled();
}

void godot_icall_BindingsGenerator_SetLogPrintEnabled(BindingsGenerator p_handle, MonoBoolean p_enabled) {
	p_handle.set_log_print_enabled(p_enabled);
}

int32_t godot_icall_BindingsGenerator_GenerateCsApi(BindingsGenerator *p_handle, MonoString *p_output_dir) {
	String output_dir = GDMonoMarshal::mono_string_to_godot(p_output_dir);
	return p_handle->generate_cs_api(output_dir);
}

uint32_t godot_icall_BindingsGenerator_Version() {
	return BindingsGenerator::get_version();
}

uint32_t godot_icall_BindingsGenerator_CsGlueVersion() {
	return CS_GLUE_VERSION;
}

int32_t godot_icall_ScriptClassParser_ParseFile(MonoString *p_filepath, MonoObject *p_classes) {
	String filepath = GDMonoMarshal::mono_string_to_godot(p_filepath);

	ScriptClassParser scp;
	Error err = scp.parse_file(filepath);
	if (err == OK) {
		Array classes = GDMonoMarshal::mono_object_to_variant(p_classes);
		const Vector<ScriptClassParser::ClassDecl> &class_decls = scp.get_classes();

		for (int i = 0; i < class_decls.size(); i++) {
			const ScriptClassParser::ClassDecl &classDecl = class_decls[i];

			Dictionary classDeclDict;
			classDeclDict["name"] = classDecl.name;
			classDeclDict["namespace"] = classDecl.namespace_;
			classDeclDict["nested"] = classDecl.nested;
			classDeclDict["base_count"] = classDecl.base.size();
			classes.push_back(classDeclDict);
		}
	}
	return err;
}

uint32_t godot_icall_GodotSharpExport_GetExportedAssemblyDependencies(MonoString *p_project_dll_name, MonoString *p_project_dll_src_path,
		MonoString *p_build_config, MonoString *p_custom_lib_dir, MonoObject *r_dependencies) {
	String project_dll_name = GDMonoMarshal::mono_string_to_godot(p_project_dll_name);
	String project_dll_src_path = GDMonoMarshal::mono_string_to_godot(p_project_dll_src_path);
	String build_config = GDMonoMarshal::mono_string_to_godot(p_build_config);
	String custom_lib_dir = GDMonoMarshal::mono_string_to_godot(p_custom_lib_dir);
	Dictionary dependencies = GDMonoMarshal::mono_object_to_variant(r_dependencies);

	return GodotSharpExport::get_exported_assembly_dependencies(project_dll_name, project_dll_src_path, build_config, custom_lib_dir, dependencies);
}

float godot_icall_Internal_EditorScale() {
	return EDSCALE;
}

MonoObject *godot_icall_Internal_GlobalDef(MonoString *p_setting, MonoObject *p_default_value, MonoBoolean p_restart_if_changed) {
	String setting = GDMonoMarshal::mono_string_to_godot(p_setting);
	Variant default_value = GDMonoMarshal::mono_object_to_variant(p_default_value);
	Variant result = _GLOBAL_DEF(setting, default_value, (bool)p_restart_if_changed);
	return GDMonoMarshal::variant_to_mono_object(result);
}

MonoObject *godot_icall_Internal_EditorDef(MonoString *p_setting, MonoObject *p_default_value, MonoBoolean p_restart_if_changed) {
	String setting = GDMonoMarshal::mono_string_to_godot(p_setting);
	Variant default_value = GDMonoMarshal::mono_object_to_variant(p_default_value);
	Variant result = _GLOBAL_DEF(setting, default_value, (bool)p_restart_if_changed);
	return GDMonoMarshal::variant_to_mono_object(result);
}

MonoString *godot_icall_Internal_FullTemplatesDir() {
	String full_templates_dir = EditorSettings::get_singleton()->get_templates_dir().plus_file(VERSION_FULL_CONFIG);
	return GDMonoMarshal::mono_string_from_godot(full_templates_dir);
}

MonoString *godot_icall_Internal_SimplifyGodotPath(MonoString *p_path) {
	String path = GDMonoMarshal::mono_string_to_godot(p_path);
	return GDMonoMarshal::mono_string_from_godot(path.simplify_path());
}

MonoBoolean godot_icall_Internal_IsOsxAppBundleInstalled(MonoString *p_bundle_id) {
#ifdef OSX_ENABLED
	String bundle_id = GDMonoMarshal::mono_string_to_godot(p_bundle_id);
	return (MonoBoolean)osx_is_app_bundle_installed;
#else
	(void)p_bundle_id; // UNUSED
	return (MonoBoolean) false;
#endif
}

MonoBoolean godot_icall_Internal_MetadataIsApiAssemblyInvalidated(int32_t p_api_type) {
	return GDMono::get_singleton()->metadata_is_api_assembly_invalidated((APIAssembly::Type)p_api_type);
}

void godot_icall_Internal_MetadataSetApiAssemblyInvalidated(int32_t p_api_type, MonoBoolean p_invalidated) {
	GDMono::get_singleton()->metadata_set_api_assembly_invalidated((APIAssembly::Type)p_api_type, (bool)p_invalidated);
}

MonoBoolean godot_icall_Internal_IsMessageQueueFlushing() {
	return (MonoBoolean)MessageQueue::get_singleton()->is_flushing();
}

MonoBoolean godot_icall_Internal_GodotIs32Bits() {
	return sizeof(void *) == 4;
}

MonoBoolean godot_icall_Internal_GodotIsRealTDouble() {
#ifdef REAL_T_IS_DOUBLE
	return (MonoBoolean) true;
#else
	return (MonoBoolean) false;
#endif
}

void godot_icall_Internal_GodotMainIteration() {
	Main::iteration();
}

uint64_t godot_icall_Internal_GetCoreApiHash() {
	return ClassDB::get_api_hash(ClassDB::API_CORE);
}

uint64_t godot_icall_Internal_GetEditorApiHash() {
	return ClassDB::get_api_hash(ClassDB::API_EDITOR);
}

MonoBoolean godot_icall_Internal_IsAssembliesReloadingNeeded() {
#ifdef GD_MONO_HOT_RELOAD
	return (MonoBoolean)CSharpLanguage::get_singleton()->is_assembly_reloading_needed();
#else
	return (MonoBoolean) false;
#endif
}

void godot_icall_Internal_ReloadAssemblies(MonoBoolean p_soft_reload) {
#ifdef GD_MONO_HOT_RELOAD
	_GodotSharp::get_singleton()->call_deferred("_reload_assemblies", (bool)p_soft_reload);
#endif
}

void godot_icall_Internal_ScriptEditorDebuggerReloadScripts() {
	ScriptEditor::get_singleton()->get_debugger()->reload_scripts();
}

MonoBoolean godot_icall_Internal_ScriptEditorEdit(MonoObject *p_resource, int32_t p_line, int32_t p_col, MonoBoolean p_grab_focus) {
	Ref<Resource> resource = GDMonoMarshal::mono_object_to_variant(p_resource);
	return (MonoBoolean)ScriptEditor::get_singleton()->edit(resource, p_line, p_col, (bool)p_grab_focus);
}

void godot_icall_Internal_EditorNodeShowScriptScreen() {
	EditorNode::get_singleton()->call("_editor_select", EditorNode::EDITOR_SCRIPT);
}

MonoObject *godot_icall_Internal_GetScriptsMetadataOrNothing(MonoReflectionType *p_dict_reftype) {
	Dictionary maybe_metadata = CSharpLanguage::get_singleton()->get_scripts_metadata_or_nothing();

	MonoType *dict_type = mono_reflection_type_get_type(p_dict_reftype);

	uint32_t type_encoding = mono_type_get_type(dict_type);
	MonoClass *type_class_raw = mono_class_from_mono_type(dict_type);
	GDMonoClass *type_class = GDMono::get_singleton()->get_class(type_class_raw);

	return GDMonoMarshal::variant_to_mono_object(maybe_metadata, ManagedType(type_encoding, type_class));
}

MonoString *godot_icall_Internal_MonoWindowsInstallRoot() {
#ifdef WINDOWS_ENABLED
	String install_root_dir = GDMono::get_singleton()->get_mono_reg_info().install_root_dir;
	return GDMonoMarshal::mono_string_from_godot(install_root_dir);
#else
	return NULL;
#endif
}

MonoString *godot_icall_Utils_OS_GetPlatformName() {
	String os_name = OS::get_singleton()->get_name();
	return GDMonoMarshal::mono_string_from_godot(os_name);
}

void register_editor_internal_calls() {

	// GodotSharpDirs
	mono_add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_ResDataDir", (void *)godot_icall_GodotSharpDirs_ResDataDir);
	mono_add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_ResMetadataDir", (void *)godot_icall_GodotSharpDirs_ResMetadataDir);
	mono_add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_ResAssembliesBaseDir", (void *)godot_icall_GodotSharpDirs_ResAssembliesBaseDir);
	mono_add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_ResAssembliesDir", (void *)godot_icall_GodotSharpDirs_ResAssembliesDir);
	mono_add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_ResConfigDir", (void *)godot_icall_GodotSharpDirs_ResConfigDir);
	mono_add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_ResTempDir", (void *)godot_icall_GodotSharpDirs_ResTempDir);
	mono_add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_ResTempAssembliesBaseDir", (void *)godot_icall_GodotSharpDirs_ResTempAssembliesBaseDir);
	mono_add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_ResTempAssembliesDir", (void *)godot_icall_GodotSharpDirs_ResTempAssembliesDir);
	mono_add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_MonoUserDir", (void *)godot_icall_GodotSharpDirs_MonoUserDir);
	mono_add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_MonoLogsDir", (void *)godot_icall_GodotSharpDirs_MonoLogsDir);
	mono_add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_MonoSolutionsDir", (void *)godot_icall_GodotSharpDirs_MonoSolutionsDir);
	mono_add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_BuildLogsDirs", (void *)godot_icall_GodotSharpDirs_BuildLogsDirs);
	mono_add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_ProjectSlnPath", (void *)godot_icall_GodotSharpDirs_ProjectSlnPath);
	mono_add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_ProjectCsProjPath", (void *)godot_icall_GodotSharpDirs_ProjectCsProjPath);
	mono_add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_DataEditorToolsDir", (void *)godot_icall_GodotSharpDirs_DataEditorToolsDir);
	mono_add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_DataEditorPrebuiltApiDir", (void *)godot_icall_GodotSharpDirs_DataEditorPrebuiltApiDir);
	mono_add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_DataMonoEtcDir", (void *)godot_icall_GodotSharpDirs_DataMonoEtcDir);
	mono_add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_DataMonoLibDir", (void *)godot_icall_GodotSharpDirs_DataMonoLibDir);
	mono_add_internal_call("GodotTools.Internals.GodotSharpDirs::internal_DataMonoBinDir", (void *)godot_icall_GodotSharpDirs_DataMonoBinDir);

	// EditorProgress
	mono_add_internal_call("GodotTools.Internals.EditorProgress::internal_Create", (void *)godot_icall_EditorProgress_Create);
	mono_add_internal_call("GodotTools.Internals.EditorProgress::internal_Dispose", (void *)godot_icall_EditorProgress_Dispose);
	mono_add_internal_call("GodotTools.Internals.EditorProgress::internal_Step", (void *)godot_icall_EditorProgress_Step);

	// BiningsGenerator
	mono_add_internal_call("GodotTools.Internals.BindingsGenerator::internal_Ctor", (void *)godot_icall_BindingsGenerator_Ctor);
	mono_add_internal_call("GodotTools.Internals.BindingsGenerator::internal_Dtor", (void *)godot_icall_BindingsGenerator_Dtor);
	mono_add_internal_call("GodotTools.Internals.BindingsGenerator::internal_LogPrintEnabled", (void *)godot_icall_BindingsGenerator_LogPrintEnabled);
	mono_add_internal_call("GodotTools.Internals.BindingsGenerator::internal_SetLogPrintEnabled", (void *)godot_icall_BindingsGenerator_SetLogPrintEnabled);
	mono_add_internal_call("GodotTools.Internals.BindingsGenerator::internal_GenerateCsApi", (void *)godot_icall_BindingsGenerator_GenerateCsApi);
	mono_add_internal_call("GodotTools.Internals.BindingsGenerator::internal_Version", (void *)godot_icall_BindingsGenerator_Version);
	mono_add_internal_call("GodotTools.Internals.BindingsGenerator::internal_CsGlueVersion", (void *)godot_icall_BindingsGenerator_CsGlueVersion);

	// ScriptClassParser
	mono_add_internal_call("GodotTools.Internals.ScriptClassParser::internal_ParseFile", (void *)godot_icall_ScriptClassParser_ParseFile);

	// GodotSharpExport
	mono_add_internal_call("GodotTools.GodotSharpExport::internal_GetExportedAssemblyDependencies", (void *)godot_icall_GodotSharpExport_GetExportedAssemblyDependencies);

	// Internals
	mono_add_internal_call("GodotTools.Internals.Internal::internal_EditorScale", (void *)godot_icall_Internal_EditorScale);
	mono_add_internal_call("GodotTools.Internals.Internal::internal_GlobalDef", (void *)godot_icall_Internal_GlobalDef);
	mono_add_internal_call("GodotTools.Internals.Internal::internal_EditorDef", (void *)godot_icall_Internal_EditorDef);
	mono_add_internal_call("GodotTools.Internals.Internal::internal_FullTemplatesDir", (void *)godot_icall_Internal_FullTemplatesDir);
	mono_add_internal_call("GodotTools.Internals.Internal::internal_SimplifyGodotPath", (void *)godot_icall_Internal_SimplifyGodotPath);
	mono_add_internal_call("GodotTools.Internals.Internal::internal_IsOsxAppBundleInstalled", (void *)godot_icall_Internal_IsOsxAppBundleInstalled);
	mono_add_internal_call("GodotTools.Internals.Internal::internal_MetadataIsApiAssemblyInvalidated", (void *)godot_icall_Internal_MetadataIsApiAssemblyInvalidated);
	mono_add_internal_call("GodotTools.Internals.Internal::internal_MetadataSetApiAssemblyInvalidated", (void *)godot_icall_Internal_MetadataSetApiAssemblyInvalidated);
	mono_add_internal_call("GodotTools.Internals.Internal::internal_IsMessageQueueFlushing", (void *)godot_icall_Internal_IsMessageQueueFlushing);
	mono_add_internal_call("GodotTools.Internals.Internal::internal_GodotIs32Bits", (void *)godot_icall_Internal_GodotIs32Bits);
	mono_add_internal_call("GodotTools.Internals.Internal::internal_GodotIsRealTDouble", (void *)godot_icall_Internal_GodotIsRealTDouble);
	mono_add_internal_call("GodotTools.Internals.Internal::internal_GodotMainIteration", (void *)godot_icall_Internal_GodotMainIteration);
	mono_add_internal_call("GodotTools.Internals.Internal::internal_GetCoreApiHash", (void *)godot_icall_Internal_GetCoreApiHash);
	mono_add_internal_call("GodotTools.Internals.Internal::internal_GetEditorApiHash", (void *)godot_icall_Internal_GetEditorApiHash);
	mono_add_internal_call("GodotTools.Internals.Internal::internal_IsAssembliesReloadingNeeded", (void *)godot_icall_Internal_IsAssembliesReloadingNeeded);
	mono_add_internal_call("GodotTools.Internals.Internal::internal_ReloadAssemblies", (void *)godot_icall_Internal_ReloadAssemblies);
	mono_add_internal_call("GodotTools.Internals.Internal::internal_ScriptEditorDebuggerReloadScripts", (void *)godot_icall_Internal_ScriptEditorDebuggerReloadScripts);
	mono_add_internal_call("GodotTools.Internals.Internal::internal_ScriptEditorEdit", (void *)godot_icall_Internal_ScriptEditorEdit);
	mono_add_internal_call("GodotTools.Internals.Internal::internal_EditorNodeShowScriptScreen", (void *)godot_icall_Internal_EditorNodeShowScriptScreen);
	mono_add_internal_call("GodotTools.Internals.Internal::internal_GetScriptsMetadataOrNothing", (void *)godot_icall_Internal_GetScriptsMetadataOrNothing);
	mono_add_internal_call("GodotTools.Internals.Internal::internal_MonoWindowsInstallRoot", (void *)godot_icall_Internal_MonoWindowsInstallRoot);

	// Utils.OS
	mono_add_internal_call("GodotTools.Utils.OS::GetPlatformName", (void *)godot_icall_Utils_OS_GetPlatformName);
}
