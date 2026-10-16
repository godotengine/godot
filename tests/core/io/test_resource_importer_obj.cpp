/**************************************************************************/
/*  test_resource_importer_obj.cpp                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM OUT OF OR IN CONNECTION WITH THE       */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_resource_importer_obj)

#include "core/io/dir_access.h"
#include "core/os/os.h"
#include "tests/test_utils.h"

#ifdef TOOLS_ENABLED
#include "editor/import/3d/resource_importer_obj.h"
#endif

namespace TestResourceImporterOBJ {

#ifdef TOOLS_ENABLED
static HashMap<StringName, Variant> _get_default_import_options() {
	HashMap<StringName, Variant> options;
	options["generate_tangents"] = true;
	options["generate_lods"] = true;
	options["generate_shadow_mesh"] = true;
	options["generate_lightmap_uv2"] = false;
	options["generate_lightmap_uv2_texel_size"] = 0.2f;
	options["scale_mesh"] = Vector3(1, 1, 1);
	options["offset_mesh"] = Vector3(0, 0, 0);
	options["force_disable_mesh_compression"] = false;
	return options;
}
#endif

TEST_CASE("[Editor][ResourceImporterOBJ] Import well-formed and malformed OBJ from directory") {
#ifdef TOOLS_ENABLED
	String root = OS::get_singleton()->get_environment("GODOT_OBJ_IMPORT_TEST_DIR");
	REQUIRE_MESSAGE(root.length() > 0,
			"Set GODOT_OBJ_IMPORT_TEST_DIR to a directory containing 'valid' and 'invalid' subdirs with .obj files. "
			"Run: GODOT_OBJ_IMPORT_TEST_DIR=/path/to/dir ./godot --test --test-case=\"*ResourceImporterOBJ*\"");

	Ref<ResourceImporterOBJ> importer;
	importer.instantiate();
	HashMap<StringName, Variant> options = _get_default_import_options();
	String save_base = TestUtils::get_temp_path("obj_import_test");
	Ref<DirAccess> da = DirAccess::open(save_base);
	if (da.is_valid()) {
		da->erase_contents_recursive();
	} else {
		DirAccess::make_dir_recursive_absolute(save_base);
	}

	Ref<DirAccess> root_da = DirAccess::open(root);
	REQUIRE_MESSAGE(root_da.is_valid(), "GODOT_OBJ_IMPORT_TEST_DIR must be a readable directory.");

	for (const String &subdir : { "valid", "invalid" }) {
		bool expect_ok = (subdir == "valid");
		String sub_path = root.path_join(subdir);
		Ref<DirAccess> sub_da = DirAccess::open(sub_path);
		if (sub_da.is_null()) {
			continue;
		}
		PackedStringArray files = sub_da->get_files();
		for (const String &f : files) {
			if (!f.ends_with(".obj")) {
				continue;
			}
			String src = sub_path.path_join(f);
			String base_name = f.get_basename();
			String save_path = save_base.path_join(subdir + "_" + base_name);

			List<String> gen_files;
			ERR_PRINT_OFF;
			Error err = importer->import(ResourceUID::ID(), src, save_path, options, nullptr, &gen_files, nullptr);
			ERR_PRINT_ON;

			if (expect_ok) {
				CHECK_MESSAGE(err == OK, "Valid OBJ '", f, "' should import successfully, got err ", (int64_t)err);
			} else {
				CHECK_MESSAGE(err != OK, "Malformed OBJ '", f, "' should fail import, got OK.");
			}
		}
	}
#else
	SKIP("OBJ importer tests require TOOLS_ENABLED.");
#endif
}

} // namespace TestResourceImporterOBJ
