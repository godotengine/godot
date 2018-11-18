/*************************************************************************/
/*  godotsharp_export.cpp                                                */
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

#include "godotsharp_export.h"

#include "core/version.h"

#include "../csharp_script.h"
#include "../godotsharp_defs.h"
#include "../godotsharp_dirs.h"
#include "../mono_gd/gd_mono_class.h"
#include "../mono_gd/gd_mono_marshal.h"
#include "csharp_project.h"
#include "godotsharp_builds.h"

static MonoString *godot_icall_GodotSharpExport_GetTemplatesDir() {
	String current_version = VERSION_FULL_CONFIG;
	String templates_dir = EditorSettings::get_singleton()->get_templates_dir().plus_file(current_version);
	return GDMonoMarshal::mono_string_from_godot(ProjectSettings::get_singleton()->globalize_path(templates_dir));
}

static MonoString *godot_icall_GodotSharpExport_GetDataDirName() {
	String appname = ProjectSettings::get_singleton()->get("application/config/name");
	String appname_safe = OS::get_singleton()->get_safe_dir_name(appname);
	return GDMonoMarshal::mono_string_from_godot("data_" + appname_safe);
}

void GodotSharpExport::register_internal_calls() {
	static bool registered = false;
	ERR_FAIL_COND(registered);
	registered = true;

	mono_add_internal_call("GodotSharpTools.Editor.GodotSharpExport::GetTemplatesDir", (void *)godot_icall_GodotSharpExport_GetTemplatesDir);
	mono_add_internal_call("GodotSharpTools.Editor.GodotSharpExport::GetDataDirName", (void *)godot_icall_GodotSharpExport_GetDataDirName);
}

void GodotSharpExport::_export_file(const String &p_path, const String &p_type, const Set<String> &) {

	if (p_type != CSharpLanguage::get_singleton()->get_type())
		return;

	ERR_FAIL_COND(p_path.get_extension() != CSharpLanguage::get_singleton()->get_extension());

	// TODO what if the source file is not part of the game's C# project

	if (!GLOBAL_GET("mono/export/include_scripts_content")) {
		// We don't want to include the source code on exported games
		add_file(p_path, Vector<uint8_t>(), false);
		skip();
	}
}

void GodotSharpExport::_export_begin(const Set<String> &p_features, bool p_debug, const String &p_path, int p_flags) {

	// TODO right now there is no way to stop the export process with an error

	ERR_FAIL_COND(!GDMono::get_singleton()->is_runtime_initialized());
	ERR_FAIL_NULL(TOOLS_DOMAIN);
	ERR_FAIL_NULL(GDMono::get_singleton()->get_editor_tools_assembly());

	String build_config = p_debug ? "Debug" : "Release";

	String scripts_metadata_path = GodotSharpDirs::get_res_metadata_dir().plus_file("scripts_metadata." + String(p_debug ? "debug" : "release"));
	Error metadata_err = CSharpProject::generate_scripts_metadata(GodotSharpDirs::get_project_csproj_path(), scripts_metadata_path);
	ERR_FAIL_COND(metadata_err != OK);

	ERR_FAIL_COND(!_add_file(scripts_metadata_path, scripts_metadata_path));

	ERR_FAIL_COND(!GodotSharpBuilds::build_project_blocking(build_config));

	// Add dependency assemblies

	Map<String, String> dependencies;

	String project_dll_name = ProjectSettings::get_singleton()->get("application/config/name");
	if (project_dll_name.empty()) {
		project_dll_name = "UnnamedProject";
	}

	String project_dll_src_dir = GodotSharpDirs::get_res_temp_assemblies_base_dir().plus_file(build_config);
	String project_dll_src_path = project_dll_src_dir.plus_file(project_dll_name + ".dll");
	dependencies.insert(project_dll_name, project_dll_src_path);

	{
		MonoDomain *export_domain = GDMonoUtils::create_domain("GodotEngine.ProjectExportDomain");
		ERR_FAIL_NULL(export_domain);
		_GDMONO_SCOPE_EXIT_DOMAIN_UNLOAD_(export_domain);

		_GDMONO_SCOPE_DOMAIN_(export_domain);

		GDMonoAssembly *scripts_assembly = NULL;
		bool load_success = GDMono::get_singleton()->load_assembly_from(project_dll_name,
				project_dll_src_path, &scripts_assembly, /* refonly: */ true);

		ERR_EXPLAIN("Cannot load refonly assembly: " + project_dll_name);
		ERR_FAIL_COND(!load_success);

		Vector<String> search_dirs;
		GDMonoAssembly::fill_search_dirs(search_dirs);
		Error depend_error = _get_assembly_dependencies(scripts_assembly, search_dirs, dependencies);
		ERR_FAIL_COND(depend_error != OK);
	}

	for (Map<String, String>::Element *E = dependencies.front(); E; E = E->next()) {
		String depend_src_path = E->value();
		String depend_dst_path = GodotSharpDirs::get_res_assemblies_dir().plus_file(depend_src_path.get_file());
		ERR_FAIL_COND(!_add_file(depend_src_path, depend_dst_path));
	}

	// Mono specific export template extras (data dir)

	GDMonoClass *export_class = GDMono::get_singleton()->get_editor_tools_assembly()->get_class("GodotSharpTools.Editor", "GodotSharpExport");
	ERR_FAIL_NULL(export_class);
	GDMonoMethod *export_begin_method = export_class->get_method("_ExportBegin", 4);
	ERR_FAIL_NULL(export_begin_method);

	MonoArray *features = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(String), p_features.size());
	int i = 0;
	for (const Set<String>::Element *E = p_features.front(); E; E = E->next()) {
		MonoString *boxed = GDMonoMarshal::mono_string_from_godot(E->get());
		mono_array_set(features, MonoString *, i, boxed);
		i++;
	}

	MonoBoolean debug = p_debug;
	MonoString *path = GDMonoMarshal::mono_string_from_godot(p_path);
	uint32_t flags = p_flags;
	void *args[4] = { features, &debug, path, &flags };
	MonoException *exc = NULL;
	export_begin_method->invoke_raw(NULL, args, &exc);

	if (exc) {
		GDMonoUtils::debug_print_unhandled_exception(exc);
		ERR_FAIL();
	}
}

bool GodotSharpExport::_add_file(const String &p_src_path, const String &p_dst_path, bool p_remap) {

	FileAccessRef f = FileAccess::open(p_src_path, FileAccess::READ);
	ERR_FAIL_COND_V(!f, false);

	Vector<uint8_t> data;
	data.resize(f->get_len());
	f->get_buffer(data.ptrw(), data.size());

	add_file(p_dst_path, data, p_remap);

	return true;
}

Error GodotSharpExport::_get_assembly_dependencies(GDMonoAssembly *p_assembly, const Vector<String> &p_search_dirs, Map<String, String> &r_dependencies) {

	MonoImage *image = p_assembly->get_image();

	for (int i = 0; i < mono_image_get_table_rows(image, MONO_TABLE_ASSEMBLYREF); i++) {
		MonoAssemblyName *ref_aname = aname_prealloc;
		mono_assembly_get_assemblyref(image, i, ref_aname);
		String ref_name = mono_assembly_name_get_name(ref_aname);

		if (r_dependencies.find(ref_name))
			continue;

		GDMonoAssembly *ref_assembly = NULL;
		String path;
		bool has_extension = ref_name.ends_with(".dll") || ref_name.ends_with(".exe");

		for (int i = 0; i < p_search_dirs.size(); i++) {
			const String &search_dir = p_search_dirs[i];

			if (has_extension) {
				path = search_dir.plus_file(ref_name);
				if (FileAccess::exists(path)) {
					GDMono::get_singleton()->load_assembly_from(ref_name.get_basename(), path, &ref_assembly, true);
					if (ref_assembly != NULL)
						break;
				}
			} else {
				path = search_dir.plus_file(ref_name + ".dll");
				if (FileAccess::exists(path)) {
					GDMono::get_singleton()->load_assembly_from(ref_name, path, &ref_assembly, true);
					if (ref_assembly != NULL)
						break;
				}

				path = search_dir.plus_file(ref_name + ".exe");
				if (FileAccess::exists(path)) {
					GDMono::get_singleton()->load_assembly_from(ref_name, path, &ref_assembly, true);
					if (ref_assembly != NULL)
						break;
				}
			}
		}

		if (!ref_assembly) {
			ERR_EXPLAIN("Cannot load assembly (refonly): " + ref_name);
			ERR_FAIL_V(ERR_CANT_RESOLVE);
		}

		r_dependencies.insert(ref_name, ref_assembly->get_path());

		Error err = _get_assembly_dependencies(ref_assembly, p_search_dirs, r_dependencies);
		if (err != OK)
			return err;
	}

	return OK;
}

GodotSharpExport::GodotSharpExport() {
	// MonoAssemblyName is an incomplete type (internal to mono), so we can't allocate it ourselves.
	// There isn't any api to allocate an empty one either, so we need to do it this way.
	aname_prealloc = mono_assembly_name_new("whatever");
	mono_assembly_name_free(aname_prealloc); // "it does not frees the object itself, only the name members" (typo included)
}

GodotSharpExport::~GodotSharpExport() {
	if (aname_prealloc)
		mono_free(aname_prealloc);
}
