/*************************************************************************/
/*  godotsharp_builds.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "godotsharp_builds.h"

#include "core/vector.h"
#include "main/main.h"

#include "../glue/cs_glue_version.gen.h"
#include "../godotsharp_dirs.h"
#include "../mono_gd/gd_mono_class.h"
#include "../mono_gd/gd_mono_marshal.h"
#include "../utils/path_utils.h"
#include "bindings_generator.h"
#include "csharp_project.h"
#include "godotsharp_editor.h"

#define PROP_NAME_MSBUILD_MONO "MSBuild (Mono)"
#define PROP_NAME_MSBUILD_VS "MSBuild (VS Build Tools)"
#define PROP_NAME_XBUILD "xbuild (Deprecated)"

void godot_icall_BuildInstance_ExitCallback(MonoString *p_solution, MonoString *p_config, int p_exit_code) {

	String solution = GDMonoMarshal::mono_string_to_godot(p_solution);
	String config = GDMonoMarshal::mono_string_to_godot(p_config);
	GodotSharpBuilds::get_singleton()->build_exit_callback(MonoBuildInfo(solution, config), p_exit_code);
}

static Vector<const char *> _get_msbuild_hint_dirs() {
	Vector<const char *> ret;
#ifdef OSX_ENABLED
	ret.push_back("/Library/Frameworks/Mono.framework/Versions/Current/bin/");
	ret.push_back("/usr/local/var/homebrew/linked/mono/bin/");
#endif
	ret.push_back("/opt/novell/mono/bin/");
	return ret;
}

#ifdef UNIX_ENABLED
String _find_build_engine_on_unix(const String &p_name) {
	String ret = path_which(p_name);

	if (ret.length())
		return ret;

	String ret_fallback = path_which(p_name + ".exe");
	if (ret_fallback.length())
		return ret_fallback;

	static Vector<const char *> locations = _get_msbuild_hint_dirs();

	for (int i = 0; i < locations.size(); i++) {
		String hint_path = locations[i] + p_name;

		if (FileAccess::exists(hint_path)) {
			return hint_path;
		}
	}

	return String();
}
#endif

MonoString *godot_icall_BuildInstance_get_MSBuildPath() {

	GodotSharpBuilds::BuildTool build_tool = GodotSharpBuilds::BuildTool(int(EditorSettings::get_singleton()->get("mono/builds/build_tool")));

#if defined(WINDOWS_ENABLED)
	switch (build_tool) {
		case GodotSharpBuilds::MSBUILD_VS: {
			static String msbuild_tools_path;

			if (msbuild_tools_path.empty() || !FileAccess::exists(msbuild_tools_path)) {
				// Try to search it again if it wasn't found last time or if it was removed from its location
				msbuild_tools_path = MonoRegUtils::find_msbuild_tools_path();
			}

			if (msbuild_tools_path.length()) {
				if (!msbuild_tools_path.ends_with("\\"))
					msbuild_tools_path += "\\";

				return GDMonoMarshal::mono_string_from_godot(msbuild_tools_path + "MSBuild.exe");
			}

			print_verbose("Cannot find executable for '" PROP_NAME_MSBUILD_VS "'. Trying with '" PROP_NAME_MSBUILD_MONO "'...");
		} // FALL THROUGH
		case GodotSharpBuilds::MSBUILD_MONO: {
			String msbuild_path = GDMono::get_singleton()->get_mono_reg_info().bin_dir.plus_file("msbuild.bat");

			if (!FileAccess::exists(msbuild_path)) {
				WARN_PRINTS("Cannot find executable for '" PROP_NAME_MSBUILD_MONO "'. Tried with path: " + msbuild_path);
			}

			return GDMonoMarshal::mono_string_from_godot(msbuild_path);
		} break;
		case GodotSharpBuilds::XBUILD: {
			String xbuild_path = GDMono::get_singleton()->get_mono_reg_info().bin_dir.plus_file("xbuild.bat");

			if (!FileAccess::exists(xbuild_path)) {
				WARN_PRINTS("Cannot find executable for '" PROP_NAME_XBUILD "'. Tried with path: " + xbuild_path);
			}

			return GDMonoMarshal::mono_string_from_godot(xbuild_path);
		} break;
		default:
			ERR_EXPLAIN("You don't deserve to live");
			CRASH_NOW();
	}
#elif defined(UNIX_ENABLED)
	static String msbuild_path;
	static String xbuild_path;

	if (build_tool == GodotSharpBuilds::XBUILD) {
		if (xbuild_path.empty() || !FileAccess::exists(xbuild_path)) {
			// Try to search it again if it wasn't found last time or if it was removed from its location
			xbuild_path = _find_build_engine_on_unix("msbuild");
		}

		if (xbuild_path.empty()) {
			WARN_PRINT("Cannot find binary for '" PROP_NAME_XBUILD "'");
			return NULL;
		}
	} else {
		if (msbuild_path.empty() || !FileAccess::exists(msbuild_path)) {
			// Try to search it again if it wasn't found last time or if it was removed from its location
			msbuild_path = _find_build_engine_on_unix("msbuild");
		}

		if (msbuild_path.empty()) {
			WARN_PRINT("Cannot find binary for '" PROP_NAME_MSBUILD_MONO "'");
			return NULL;
		}
	}

	return GDMonoMarshal::mono_string_from_godot(build_tool != GodotSharpBuilds::XBUILD ? msbuild_path : xbuild_path);
#else
	(void)build_tool; // UNUSED

	ERR_EXPLAIN("Not implemented on this platform");
	ERR_FAIL_V(NULL);
#endif
}

MonoString *godot_icall_BuildInstance_get_FrameworkPath() {

#if defined(WINDOWS_ENABLED)
	const MonoRegInfo &mono_reg_info = GDMono::get_singleton()->get_mono_reg_info();
	if (mono_reg_info.assembly_dir.length()) {
		String framework_path = path_join(mono_reg_info.assembly_dir, "mono", "4.5");
		return GDMonoMarshal::mono_string_from_godot(framework_path);
	}

	ERR_EXPLAIN("Cannot find Mono's assemblies directory in the registry");
	ERR_FAIL_V(NULL);
#else
	return NULL;
#endif
}

MonoString *godot_icall_BuildInstance_get_MonoWindowsBinDir() {

#if defined(WINDOWS_ENABLED)
	const MonoRegInfo &mono_reg_info = GDMono::get_singleton()->get_mono_reg_info();
	if (mono_reg_info.bin_dir.length()) {
		return GDMonoMarshal::mono_string_from_godot(mono_reg_info.bin_dir);
	}

	ERR_EXPLAIN("Cannot find Mono's binaries directory in the registry");
	ERR_FAIL_V(NULL);
#else
	return NULL;
#endif
}

MonoBoolean godot_icall_BuildInstance_get_UsingMonoMSBuildOnWindows() {

#if defined(WINDOWS_ENABLED)
	return GodotSharpBuilds::BuildTool(int(EditorSettings::get_singleton()->get("mono/builds/build_tool"))) == GodotSharpBuilds::MSBUILD_MONO;
#else
	return false;
#endif
}

void GodotSharpBuilds::register_internal_calls() {

	static bool registered = false;
	ERR_FAIL_COND(registered);
	registered = true;

	mono_add_internal_call("GodotSharpTools.Build.BuildSystem::godot_icall_BuildInstance_ExitCallback", (void *)godot_icall_BuildInstance_ExitCallback);
	mono_add_internal_call("GodotSharpTools.Build.BuildInstance::godot_icall_BuildInstance_get_MSBuildPath", (void *)godot_icall_BuildInstance_get_MSBuildPath);
	mono_add_internal_call("GodotSharpTools.Build.BuildInstance::godot_icall_BuildInstance_get_FrameworkPath", (void *)godot_icall_BuildInstance_get_FrameworkPath);
	mono_add_internal_call("GodotSharpTools.Build.BuildInstance::godot_icall_BuildInstance_get_MonoWindowsBinDir", (void *)godot_icall_BuildInstance_get_MonoWindowsBinDir);
	mono_add_internal_call("GodotSharpTools.Build.BuildInstance::godot_icall_BuildInstance_get_UsingMonoMSBuildOnWindows", (void *)godot_icall_BuildInstance_get_UsingMonoMSBuildOnWindows);
}

void GodotSharpBuilds::show_build_error_dialog(const String &p_message) {

	GodotSharpEditor::get_singleton()->show_error_dialog(p_message, "Build error");
	MonoBottomPanel::get_singleton()->show_build_tab();
}

bool GodotSharpBuilds::build_api_sln(const String &p_api_sln_dir, const String &p_config) {

	String api_sln_file = p_api_sln_dir.plus_file(API_SOLUTION_NAME ".sln");

	String core_api_assembly_dir = p_api_sln_dir.plus_file(CORE_API_ASSEMBLY_NAME).plus_file("bin").plus_file(p_config);
	String core_api_assembly_file = core_api_assembly_dir.plus_file(CORE_API_ASSEMBLY_NAME ".dll");

	String editor_api_assembly_dir = p_api_sln_dir.plus_file(EDITOR_API_ASSEMBLY_NAME).plus_file("bin").plus_file(p_config);
	String editor_api_assembly_file = editor_api_assembly_dir.plus_file(EDITOR_API_ASSEMBLY_NAME ".dll");

	if (!FileAccess::exists(core_api_assembly_file) || !FileAccess::exists(editor_api_assembly_file)) {
		MonoBuildInfo api_build_info(api_sln_file, p_config);
		// TODO Replace this global NoWarn with '#pragma warning' directives on generated files,
		// once we start to actively document manually maintained C# classes
		api_build_info.custom_props.push_back("NoWarn=1591"); // Ignore missing documentation warnings

		if (!GodotSharpBuilds::get_singleton()->build(api_build_info)) {
			show_build_error_dialog("Failed to build " API_SOLUTION_NAME " solution.");
			return false;
		}
	}

	return true;
}

bool GodotSharpBuilds::copy_api_assembly(const String &p_src_dir, const String &p_dst_dir, const String &p_assembly_name, APIAssembly::Type p_api_type) {

	// Create destination directory if needed
	if (!DirAccess::exists(p_dst_dir)) {
		DirAccess *da = DirAccess::create_for_path(p_dst_dir);
		Error err = da->make_dir_recursive(p_dst_dir);
		memdelete(da);

		if (err != OK) {
			show_build_error_dialog("Failed to create destination directory for the API assemblies. Error: " + itos(err));
			return false;
		}
	}

	String assembly_file = p_assembly_name + ".dll";
	String assembly_src = p_src_dir.plus_file(assembly_file);
	String assembly_dst = p_dst_dir.plus_file(assembly_file);

	if (!FileAccess::exists(assembly_dst) ||
			FileAccess::get_modified_time(assembly_src) > FileAccess::get_modified_time(assembly_dst) ||
			GDMono::get_singleton()->metadata_is_api_assembly_invalidated(p_api_type)) {
		DirAccessRef da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);

		String xml_file = p_assembly_name + ".xml";
		if (da->copy(p_src_dir.plus_file(xml_file), p_dst_dir.plus_file(xml_file)) != OK)
			WARN_PRINTS("Failed to copy " + xml_file);

		String pdb_file = p_assembly_name + ".pdb";
		if (da->copy(p_src_dir.plus_file(pdb_file), p_dst_dir.plus_file(pdb_file)) != OK)
			WARN_PRINTS("Failed to copy " + pdb_file);

		Error err = da->copy(assembly_src, assembly_dst);

		if (err != OK) {
			show_build_error_dialog("Failed to copy " + assembly_file);
			return false;
		}

		GDMono::get_singleton()->metadata_set_api_assembly_invalidated(p_api_type, false);
	}

	return true;
}

String GodotSharpBuilds::_api_folder_name(APIAssembly::Type p_api_type) {

	uint64_t api_hash = p_api_type == APIAssembly::API_CORE ?
								GDMono::get_singleton()->get_api_core_hash() :
								GDMono::get_singleton()->get_api_editor_hash();
	return String::num_uint64(api_hash) +
		   "_" + String::num_uint64(BindingsGenerator::get_version()) +
		   "_" + String::num_uint64(CS_GLUE_VERSION);
}

bool GodotSharpBuilds::make_api_assembly(APIAssembly::Type p_api_type) {

	String api_name = p_api_type == APIAssembly::API_CORE ? CORE_API_ASSEMBLY_NAME : EDITOR_API_ASSEMBLY_NAME;

	String editor_prebuilt_api_dir = GodotSharpDirs::get_data_editor_prebuilt_api_dir();
	String res_assemblies_dir = GodotSharpDirs::get_res_assemblies_dir();

	if (FileAccess::exists(editor_prebuilt_api_dir.plus_file(api_name + ".dll"))) {
		EditorProgress pr("mono_copy_prebuilt_api_assembly", "Copying prebuilt " + api_name + " assembly...", 1);
		pr.step("Copying " + api_name + " assembly", 0);
		return GodotSharpBuilds::copy_api_assembly(editor_prebuilt_api_dir, res_assemblies_dir, api_name, p_api_type);
	}

	String api_build_config = "Release";

	EditorProgress pr("mono_build_release_" API_SOLUTION_NAME, "Building " API_SOLUTION_NAME " solution...", 3);

	pr.step("Generating " API_SOLUTION_NAME " solution", 0);

	String api_sln_dir = GodotSharpDirs::get_mono_solutions_dir()
								 .plus_file(_api_folder_name(APIAssembly::API_CORE));

	String api_sln_file = api_sln_dir.plus_file(API_SOLUTION_NAME ".sln");

	if (!DirAccess::exists(api_sln_dir) || !FileAccess::exists(api_sln_file)) {
		BindingsGenerator *gen = BindingsGenerator::get_singleton();
		bool gen_verbose = OS::get_singleton()->is_stdout_verbose();

		Error err = gen->generate_cs_api(api_sln_dir, gen_verbose);
		if (err != OK) {
			show_build_error_dialog("Failed to generate " API_SOLUTION_NAME " solution. Error: " + itos(err));
			return false;
		}
	}

	pr.step("Building " API_SOLUTION_NAME " solution", 1);

	if (!GodotSharpBuilds::build_api_sln(api_sln_dir, api_build_config))
		return false;

	pr.step("Copying " + api_name + " assembly", 2);

	// Copy the built assembly to the assemblies directory
	String api_assembly_dir = api_sln_dir.plus_file(api_name).plus_file("bin").plus_file(api_build_config);
	if (!GodotSharpBuilds::copy_api_assembly(api_assembly_dir, res_assemblies_dir, api_name, p_api_type))
		return false;

	return true;
}

bool GodotSharpBuilds::build_project_blocking(const String &p_config) {

	if (!FileAccess::exists(GodotSharpDirs::get_project_sln_path()))
		return true; // No solution to build

	if (!GodotSharpBuilds::make_api_assembly(APIAssembly::API_CORE))
		return false;

	if (!GodotSharpBuilds::make_api_assembly(APIAssembly::API_EDITOR))
		return false;

	EditorProgress pr("mono_project_debug_build", "Building project solution...", 1);
	pr.step("Building project solution", 0);

	MonoBuildInfo build_info(GodotSharpDirs::get_project_sln_path(), p_config);
	if (!GodotSharpBuilds::get_singleton()->build(build_info)) {
		GodotSharpBuilds::show_build_error_dialog("Failed to build project solution");
		return false;
	}

	return true;
}

bool GodotSharpBuilds::editor_build_callback() {

	String scripts_metadata_path_editor = GodotSharpDirs::get_res_metadata_dir().plus_file("scripts_metadata.editor");
	String scripts_metadata_path_player = GodotSharpDirs::get_res_metadata_dir().plus_file("scripts_metadata.editor_player");

	Error metadata_err = CSharpProject::generate_scripts_metadata(GodotSharpDirs::get_project_csproj_path(), scripts_metadata_path_editor);
	ERR_FAIL_COND_V(metadata_err != OK, false);

	if (FileAccess::exists(scripts_metadata_path_editor)) {
		DirAccessRef da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
		Error copy_err = da->copy(scripts_metadata_path_editor, scripts_metadata_path_player);

		ERR_EXPLAIN("Failed to copy scripts metadata file");
		ERR_FAIL_COND_V(copy_err != OK, false);
	}

	return build_project_blocking("Tools");
}

GodotSharpBuilds *GodotSharpBuilds::singleton = NULL;

void GodotSharpBuilds::build_exit_callback(const MonoBuildInfo &p_build_info, int p_exit_code) {

	BuildProcess *match = builds.getptr(p_build_info);
	ERR_FAIL_NULL(match);

	BuildProcess &bp = *match;
	bp.on_exit(p_exit_code);
}

void GodotSharpBuilds::restart_build(MonoBuildTab *p_build_tab) {
}

void GodotSharpBuilds::stop_build(MonoBuildTab *p_build_tab) {
}

bool GodotSharpBuilds::build(const MonoBuildInfo &p_build_info) {

	BuildProcess *match = builds.getptr(p_build_info);

	if (match) {
		BuildProcess &bp = *match;
		bp.start(true);
		return bp.exit_code == 0;
	} else {
		BuildProcess bp = BuildProcess(p_build_info);
		bp.start(true);
		builds.set(p_build_info, bp);
		return bp.exit_code == 0;
	}
}

bool GodotSharpBuilds::build_async(const MonoBuildInfo &p_build_info, GodotSharpBuild_ExitCallback p_callback) {

	BuildProcess *match = builds.getptr(p_build_info);

	if (match) {
		BuildProcess &bp = *match;
		bp.start();
		return !bp.exited; // failed to start
	} else {
		BuildProcess bp = BuildProcess(p_build_info, p_callback);
		bp.start();
		builds.set(p_build_info, bp);
		return !bp.exited; // failed to start
	}
}

GodotSharpBuilds::GodotSharpBuilds() {

	singleton = this;

	EditorNode::get_singleton()->add_build_callback(&GodotSharpBuilds::editor_build_callback);

	// Build tool settings
	EditorSettings *ed_settings = EditorSettings::get_singleton();

	EDITOR_DEF("mono/builds/build_tool", MSBUILD_MONO);

	ed_settings->add_property_hint(PropertyInfo(Variant::INT, "mono/builds/build_tool", PROPERTY_HINT_ENUM,
			PROP_NAME_MSBUILD_MONO
#ifdef WINDOWS_ENABLED
			"," PROP_NAME_MSBUILD_VS
#endif
			"," PROP_NAME_XBUILD));
}

GodotSharpBuilds::~GodotSharpBuilds() {

	singleton = NULL;
}

void GodotSharpBuilds::BuildProcess::on_exit(int p_exit_code) {

	exited = true;
	exit_code = p_exit_code;
	build_tab->on_build_exit(p_exit_code == 0 ? MonoBuildTab::RESULT_SUCCESS : MonoBuildTab::RESULT_ERROR);
	build_instance.unref();

	if (exit_callback)
		exit_callback(exit_code);
}

void GodotSharpBuilds::BuildProcess::start(bool p_blocking) {

	_GDMONO_SCOPE_DOMAIN_(TOOLS_DOMAIN)

	exit_code = -1;

	String log_dirpath = build_info.get_log_dirpath();

	if (build_tab) {
		build_tab->on_build_start();
	} else {
		build_tab = memnew(MonoBuildTab(build_info, log_dirpath));
		MonoBottomPanel::get_singleton()->add_build_tab(build_tab);
	}

	if (p_blocking) {
		// Required in order to update the build tasks list
		Main::iteration();
	}

	if (!exited) {
		exited = true;
		String message = "Tried to start build process, but it is already running";
		build_tab->on_build_exec_failed(message);
		ERR_EXPLAIN(message);
		ERR_FAIL();
	}

	exited = false;

	// Remove old issues file

	String issues_file = "msbuild_issues.csv";
	DirAccessRef d = DirAccess::create_for_path(log_dirpath);
	if (d->file_exists(issues_file)) {
		Error err = d->remove(issues_file);
		if (err != OK) {
			exited = true;
			String file_path = ProjectSettings::get_singleton()->localize_path(log_dirpath).plus_file(issues_file);
			String message = "Cannot remove issues file: " + file_path;
			build_tab->on_build_exec_failed(message);
			ERR_EXPLAIN(message);
			ERR_FAIL();
		}
	}

	GDMonoClass *klass = GDMono::get_singleton()->get_editor_tools_assembly()->get_class("GodotSharpTools.Build", "BuildInstance");

	MonoObject *mono_object = mono_object_new(mono_domain_get(), klass->get_mono_ptr());

	// Construct

	Variant solution = build_info.solution;
	Variant config = build_info.configuration;

	const Variant *ctor_args[2] = { &solution, &config };

	MonoException *exc = NULL;
	GDMonoMethod *ctor = klass->get_method(".ctor", 2);
	ctor->invoke(mono_object, ctor_args, &exc);

	if (exc) {
		exited = true;
		GDMonoUtils::debug_unhandled_exception(exc);
		String message = "The build constructor threw an exception.\n" + GDMonoUtils::get_exception_name_and_message(exc);
		build_tab->on_build_exec_failed(message);
		ERR_EXPLAIN(message);
		ERR_FAIL();
	}

	// Call Build

	String logger_assembly_path = GDMono::get_singleton()->get_editor_tools_assembly()->get_path();
	Variant logger_assembly = ProjectSettings::get_singleton()->globalize_path(logger_assembly_path);
	Variant logger_output_dir = log_dirpath;
	Variant custom_props = build_info.custom_props;

	const Variant *args[3] = { &logger_assembly, &logger_output_dir, &custom_props };

	exc = NULL;
	GDMonoMethod *build_method = klass->get_method(p_blocking ? "Build" : "BuildAsync", 3);
	build_method->invoke(mono_object, args, &exc);

	if (exc) {
		exited = true;
		GDMonoUtils::debug_unhandled_exception(exc);
		String message = "The build method threw an exception.\n" + GDMonoUtils::get_exception_name_and_message(exc);
		build_tab->on_build_exec_failed(message);
		ERR_EXPLAIN(message);
		ERR_FAIL();
	}

	// Build returned

	if (p_blocking) {
		exited = true;
		exit_code = klass->get_field("exitCode")->get_int_value(mono_object);

		if (exit_code != 0) {
			print_verbose("MSBuild finished with exit code " + itos(exit_code));
		}

		build_tab->on_build_exit(exit_code == 0 ? MonoBuildTab::RESULT_SUCCESS : MonoBuildTab::RESULT_ERROR);
	} else {
		build_instance = MonoGCHandle::create_strong(mono_object);
		exited = false;
	}
}

GodotSharpBuilds::BuildProcess::BuildProcess(const MonoBuildInfo &p_build_info, GodotSharpBuild_ExitCallback p_callback) :
		build_info(p_build_info),
		build_tab(NULL),
		exit_callback(p_callback),
		exited(true),
		exit_code(-1) {
}
