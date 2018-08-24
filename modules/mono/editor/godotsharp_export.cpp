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

#include "../csharp_script.h"
#include "../godotsharp_defs.h"
#include "../godotsharp_dirs.h"
#include "godotsharp_builds.h"

void GodotSharpExport::_export_file(const String &p_path, const String &p_type, const Set<String> &p_features) {

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
	ERR_FAIL_NULL(GDMono::get_singleton()->get_tools_domain());

	String build_config = p_debug ? "Debug" : "Release";

	ERR_FAIL_COND(!GodotSharpBuilds::build_project_blocking(build_config));

	// Add API assemblies

	String core_api_dll_path = GodotSharpDirs::get_res_assemblies_dir().plus_file(API_ASSEMBLY_NAME ".dll");
	ERR_FAIL_COND(!_add_assembly(core_api_dll_path, core_api_dll_path));

	String editor_api_dll_path = GodotSharpDirs::get_res_assemblies_dir().plus_file(EDITOR_API_ASSEMBLY_NAME ".dll");
	ERR_FAIL_COND(!_add_assembly(editor_api_dll_path, editor_api_dll_path));

	// Add project assembly

	String project_dll_name = ProjectSettings::get_singleton()->get("application/config/name");
	if (project_dll_name.empty()) {
		project_dll_name = "UnnamedProject";
	}

	String project_dll_src_path = GodotSharpDirs::get_res_temp_assemblies_base_dir().plus_file(build_config).plus_file(project_dll_name + ".dll");
	String project_dll_dst_path = GodotSharpDirs::get_res_assemblies_dir().plus_file(project_dll_name + ".dll");
	ERR_FAIL_COND(!_add_assembly(project_dll_src_path, project_dll_dst_path));

	// Add dependencies

	MonoDomain *prev_domain = mono_domain_get();
	MonoDomain *export_domain = GDMonoUtils::create_domain("GodotEngine.ProjectExportDomain");

	ERR_FAIL_COND(!export_domain);
	ERR_FAIL_COND(!mono_domain_set(export_domain, false));

	Map<String, String> dependencies;
	dependencies.insert("mscorlib", GDMono::get_singleton()->get_corlib_assembly()->get_path());

	GDMonoAssembly *scripts_assembly = GDMonoAssembly::load_from(project_dll_name, project_dll_src_path, /* refonly: */ true);

	ERR_EXPLAIN("Cannot load refonly assembly: " + project_dll_name);
	ERR_FAIL_COND(!scripts_assembly);

	Error depend_error = _get_assembly_dependencies(scripts_assembly, dependencies);

	GDMono::get_singleton()->finalize_and_unload_domain(export_domain);
	mono_domain_set(prev_domain, false);

	ERR_FAIL_COND(depend_error != OK);

	for (Map<String, String>::Element *E = dependencies.front(); E; E = E->next()) {
		String depend_src_path = E->value();
		String depend_dst_path = GodotSharpDirs::get_res_assemblies_dir().plus_file(depend_src_path.get_file());
		ERR_FAIL_COND(!_add_assembly(depend_src_path, depend_dst_path));
	}
}

bool GodotSharpExport::_add_assembly(const String &p_src_path, const String &p_dst_path) {

	FileAccessRef f = FileAccess::open(p_src_path, FileAccess::READ);
	ERR_FAIL_COND_V(!f, false);

	Vector<uint8_t> data;
	data.resize(f->get_len());
	f->get_buffer(data.ptrw(), data.size());

	add_file(p_dst_path, data, false);

	return true;
}

Error GodotSharpExport::_get_assembly_dependencies(GDMonoAssembly *p_assembly, Map<String, String> &r_dependencies) {

	MonoImage *image = p_assembly->get_image();

	for (int i = 0; i < mono_image_get_table_rows(image, MONO_TABLE_ASSEMBLYREF); i++) {
		MonoAssemblyName *ref_aname = aname_prealloc;
		mono_assembly_get_assemblyref(image, i, ref_aname);
		String ref_name = mono_assembly_name_get_name(ref_aname);

		if (ref_name == "mscorlib" || r_dependencies.find(ref_name))
			continue;

		GDMonoAssembly *ref_assembly = NULL;
		if (!GDMono::get_singleton()->load_assembly(ref_name, ref_aname, &ref_assembly, /* refonly: */ true)) {
			ERR_EXPLAIN("Cannot load refonly assembly: " + ref_name);
			ERR_FAIL_V(ERR_CANT_RESOLVE);
		}

		r_dependencies.insert(ref_name, ref_assembly->get_path());

		Error err = _get_assembly_dependencies(ref_assembly, r_dependencies);
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
