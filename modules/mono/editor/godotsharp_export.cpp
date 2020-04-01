/*************************************************************************/
/*  godotsharp_export.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include <mono/metadata/image.h>

#include "core/os/os.h"

#include "../mono_gd/gd_mono.h"
#include "../mono_gd/gd_mono_assembly.h"
#include "../mono_gd/gd_mono_cache.h"

namespace GodotSharpExport {

String get_assemblyref_name(MonoImage *p_image, int index) {
	const MonoTableInfo *table_info = mono_image_get_table_info(p_image, MONO_TABLE_ASSEMBLYREF);

	uint32_t cols[MONO_ASSEMBLYREF_SIZE];

	mono_metadata_decode_row(table_info, index, cols, MONO_ASSEMBLYREF_SIZE);

	return String::utf8(mono_metadata_string_heap(p_image, cols[MONO_ASSEMBLYREF_NAME]));
}

Error get_assembly_dependencies(GDMonoAssembly *p_assembly, const Vector<String> &p_search_dirs, Dictionary &r_dependencies) {
	MonoImage *image = p_assembly->get_image();

	for (int i = 0; i < mono_image_get_table_rows(image, MONO_TABLE_ASSEMBLYREF); i++) {
		String ref_name = get_assemblyref_name(image, i);

		if (r_dependencies.has(ref_name))
			continue;

		GDMonoAssembly *ref_assembly = nullptr;
		String path;
		bool has_extension = ref_name.ends_with(".dll") || ref_name.ends_with(".exe");

		for (int j = 0; j < p_search_dirs.size(); j++) {
			const String &search_dir = p_search_dirs[j];

			if (has_extension) {
				path = search_dir.plus_file(ref_name);
				if (FileAccess::exists(path)) {
					GDMono::get_singleton()->load_assembly_from(ref_name.get_basename(), path, &ref_assembly, true);
					if (ref_assembly != nullptr)
						break;
				}
			} else {
				path = search_dir.plus_file(ref_name + ".dll");
				if (FileAccess::exists(path)) {
					GDMono::get_singleton()->load_assembly_from(ref_name, path, &ref_assembly, true);
					if (ref_assembly != nullptr)
						break;
				}

				path = search_dir.plus_file(ref_name + ".exe");
				if (FileAccess::exists(path)) {
					GDMono::get_singleton()->load_assembly_from(ref_name, path, &ref_assembly, true);
					if (ref_assembly != nullptr)
						break;
				}
			}
		}

		ERR_FAIL_COND_V_MSG(!ref_assembly, ERR_CANT_RESOLVE, "Cannot load assembly (refonly): '" + ref_name + "'.");

		// Use the path we got from the search. Don't try to get the path from the loaded assembly as we can't trust it will be from the selected BCL dir.
		r_dependencies[ref_name] = path;

		Error err = get_assembly_dependencies(ref_assembly, p_search_dirs, r_dependencies);
		ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot load one of the dependencies for the assembly: '" + ref_name + "'.");
	}

	return OK;
}

Error get_exported_assembly_dependencies(const Dictionary &p_initial_dependencies,
		const String &p_build_config, const String &p_custom_bcl_dir, Dictionary &r_dependencies) {
	MonoDomain *export_domain = GDMonoUtils::create_domain("GodotEngine.Domain.ProjectExport");
	ERR_FAIL_NULL_V(export_domain, FAILED);
	_GDMONO_SCOPE_EXIT_DOMAIN_UNLOAD_(export_domain);

	_GDMONO_SCOPE_DOMAIN_(export_domain);

	Vector<String> search_dirs;
	GDMonoAssembly::fill_search_dirs(search_dirs, p_build_config, p_custom_bcl_dir);

	for (const Variant *key = p_initial_dependencies.next(); key; key = p_initial_dependencies.next(key)) {
		String assembly_name = *key;
		String assembly_path = p_initial_dependencies[*key];

		GDMonoAssembly *assembly = nullptr;
		bool load_success = GDMono::get_singleton()->load_assembly_from(assembly_name, assembly_path, &assembly, /* refonly: */ true);

		ERR_FAIL_COND_V_MSG(!load_success, ERR_CANT_RESOLVE, "Cannot load assembly (refonly): '" + assembly_name + "'.");

		Error err = get_assembly_dependencies(assembly, search_dirs, r_dependencies);
		if (err != OK)
			return err;
	}

	return OK;
}

} // namespace GodotSharpExport
