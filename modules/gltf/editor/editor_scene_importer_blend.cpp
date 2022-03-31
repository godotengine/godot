/*************************************************************************/
/*  editor_scene_importer_blend.cpp                                      */
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

#include "editor_scene_importer_blend.h"

#if TOOLS_ENABLED

#include "../gltf_document.h"
#include "../gltf_state.h"

#include "core/config/project_settings.h"
#include "editor/editor_settings.h"
#include "scene/main/node.h"
#include "scene/resources/animation.h"

uint32_t EditorSceneFormatImporterBlend::get_import_flags() const {
	return ImportFlags::IMPORT_SCENE | ImportFlags::IMPORT_ANIMATION;
}

void EditorSceneFormatImporterBlend::get_extensions(List<String> *r_extensions) const {
	r_extensions->push_back("blend");
}

Node *EditorSceneFormatImporterBlend::import_scene(const String &p_path, uint32_t p_flags,
		const Map<StringName, Variant> &p_options, int p_bake_fps,
		List<String> *r_missing_deps, Error *r_err) {
	// Get global paths for source and sink.

	const String source_global = ProjectSettings::get_singleton()->globalize_path(p_path);
	const String sink = ProjectSettings::get_singleton()->get_imported_files_path().plus_file(
			vformat("%s-%s.gltf", p_path.get_file().get_basename(), p_path.md5_text()));
	const String sink_global = ProjectSettings::get_singleton()->globalize_path(sink);

	// Handle configuration options.

	String parameters_arg;

	if (p_options.has(SNAME("blender/nodes/custom_properties")) && p_options[SNAME("blender/nodes/custom_properties")]) {
		parameters_arg += "export_extras=True,";
	} else {
		parameters_arg += "export_extras=False,";
	}
	if (p_options.has(SNAME("blender/meshes/skins")) && p_options[SNAME("blender/meshes/skins")]) {
		int32_t skins = p_options["blender/meshes/skins"];
		if (skins == BLEND_BONE_INFLUENCES_NONE) {
			parameters_arg += "export_all_influences=False,";
		} else if (skins == BLEND_BONE_INFLUENCES_COMPATIBLE) {
			parameters_arg += "export_all_influences=False,";
		} else if (skins == BLEND_BONE_INFLUENCES_ALL) {
			parameters_arg += "export_all_influences=True,";
		}
		parameters_arg += "export_skins=True,";
	} else {
		parameters_arg += "export_skins=False,";
	}
	if (p_options.has(SNAME("blender/materials/export_materials")) && p_options[SNAME("blender/materials/export_materials")]) {
		int32_t exports = p_options["blender/materials/export_materials"];
		if (exports == BLEND_MATERIAL_EXPORT_PLACEHOLDER) {
			parameters_arg += "export_materials='PLACEHOLDER',";
		} else if (exports == BLEND_MATERIAL_EXPORT_EXPORT) {
			parameters_arg += "export_materials='EXPORT',";
		}
	} else {
		parameters_arg += "export_materials='PLACEHOLDER',";
	}
	if (p_options.has(SNAME("blender/nodes/cameras")) && p_options[SNAME("blender/nodes/cameras")]) {
		parameters_arg += "export_cameras=True,";
	} else {
		parameters_arg += "export_cameras=False,";
	}
	if (p_options.has(SNAME("blender/nodes/punctual_lights")) && p_options[SNAME("blender/nodes/punctual_lights")]) {
		parameters_arg += "export_lights=True,";
	} else {
		parameters_arg += "export_lights=False,";
	}
	if (p_options.has(SNAME("blender/meshes/colors")) && p_options[SNAME("blender/meshes/colors")]) {
		parameters_arg += "export_colors=True,";
	} else {
		parameters_arg += "export_colors=False,";
	}
	if (p_options.has(SNAME("blender/nodes/visible")) && p_options[SNAME("blender/nodes/visible")]) {
		int32_t visible = p_options["blender/nodes/visible"];
		if (visible == BLEND_VISIBLE_VISIBLE_ONLY) {
			parameters_arg += "use_visible=True,";
		} else if (visible == BLEND_VISIBLE_RENDERABLE) {
			parameters_arg += "use_renderable=True,";
		} else if (visible == BLEND_VISIBLE_ALL) {
			parameters_arg += "use_visible=False,use_renderable=False,";
		}
	} else {
		parameters_arg += "use_visible=False,use_renderable=False,";
	}

	if (p_options.has(SNAME("blender/meshes/uvs")) && p_options[SNAME("blender/meshes/uvs")]) {
		parameters_arg += "export_texcoords=True,";
	} else {
		parameters_arg += "export_texcoords=False,";
	}
	if (p_options.has(SNAME("blender/meshes/normals")) && p_options[SNAME("blender/meshes/normals")]) {
		parameters_arg += "export_normals=True,";
	} else {
		parameters_arg += "export_normals=False,";
	}
	if (p_options.has(SNAME("blender/meshes/tangents")) && p_options[SNAME("blender/meshes/tangents")]) {
		parameters_arg += "export_tangents=True,";
	} else {
		parameters_arg += "export_tangents=False,";
	}
	if (p_options.has(SNAME("blender/animation/group_tracks")) && p_options[SNAME("blender/animation/group_tracks")]) {
		parameters_arg += "export_nla_strips=True,";
	} else {
		parameters_arg += "export_nla_strips=False,";
	}
	if (p_options.has(SNAME("blender/animation/limit_playback")) && p_options[SNAME("blender/animation/limit_playback")]) {
		parameters_arg += "export_frame_range=True,";
	} else {
		parameters_arg += "export_frame_range=False,";
	}
	if (p_options.has(SNAME("blender/animation/always_sample")) && p_options[SNAME("blender/animation/always_sample")]) {
		parameters_arg += "export_force_sampling=True,";
	} else {
		parameters_arg += "export_force_sampling=False,";
	}
	if (p_options.has(SNAME("blender/meshes/export_bones_deforming_mesh_only")) && p_options[SNAME("blender/meshes/export_bones_deforming_mesh_only")]) {
		parameters_arg += "export_def_bones=True,";
	} else {
		parameters_arg += "export_def_bones=False,";
	}
	if (p_options.has(SNAME("blender/nodes/modifiers")) && p_options[SNAME("blender/nodes/modifiers")]) {
		parameters_arg += "export_apply=True";
	} else {
		parameters_arg += "export_apply=False";
	}

	String unpack_all;
	if (p_options.has(SNAME("blender/materials/unpack_enabled")) && p_options[SNAME("blender/materials/unpack_enabled")]) {
		unpack_all = "bpy.ops.file.unpack_all(method='USE_LOCAL');";
	}

	// Prepare Blender export script.

	String common_args = vformat("filepath='%s',", sink_global) +
			"export_format='GLTF_SEPARATE',"
			"export_yup=True," +
			parameters_arg;
	String script =
			String("import bpy, sys;") +
			"print('Blender 3.0 or higher is required.', file=sys.stderr) if bpy.app.version < (3, 0, 0) else None;" +
			vformat("bpy.ops.wm.open_mainfile(filepath='%s');", source_global) +
			unpack_all +
			vformat("bpy.ops.export_scene.gltf(export_keep_originals=True,%s);", common_args);
	print_verbose(script);

	// Run script with configured Blender binary.

	String blender_path = EDITOR_GET("filesystem/import/blend/blender_path");

	List<String> args;
	args.push_back("--background");
	args.push_back("--python-expr");
	args.push_back(script);

	String standard_out;
	int ret;
	OS::get_singleton()->execute(blender_path, args, &standard_out, &ret, true);
	print_verbose(blender_path);
	print_verbose(standard_out);

	if (ret != 0) {
		if (r_err) {
			*r_err = ERR_SCRIPT_FAILED;
		}
		ERR_PRINT(vformat("Blend export to glTF failed with error: %d.", ret));
		return nullptr;
	}

	// Import the generated glTF.

	// Use GLTFDocument instead of glTF importer to keep image references.
	Ref<GLTFDocument> gltf;
	gltf.instantiate();
	Ref<GLTFState> state;
	state.instantiate();
	String base_dir;
	if (p_options.has(SNAME("blender/materials/unpack_enabled")) && p_options[SNAME("blender/materials/unpack_enabled")]) {
		base_dir = sink.get_base_dir();
	}
	Error err = gltf->append_from_file(sink.get_basename() + ".gltf", state, p_flags, p_bake_fps, base_dir);
	if (err != OK) {
		if (r_err) {
			*r_err = FAILED;
		}
		return nullptr;
	}
	return gltf->generate_scene(state, p_bake_fps);
}

Ref<Animation> EditorSceneFormatImporterBlend::import_animation(const String &p_path, uint32_t p_flags,
		const Map<StringName, Variant> &p_options, int p_bake_fps) {
	return Ref<Animation>();
}

Variant EditorSceneFormatImporterBlend::get_option_visibility(const String &p_path, const String &p_option,
		const Map<StringName, Variant> &p_options) {
	if (p_option.begins_with("animation/")) {
		if (p_option != "animation/import" && !bool(p_options["animation/import"])) {
			return false;
		}
	}
	return true;
}

void EditorSceneFormatImporterBlend::get_import_options(const String &p_path, List<ResourceImporter::ImportOption> *r_options) {
#define ADD_OPTION_BOOL(PATH, VALUE) \
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, SNAME(PATH)), VALUE));
#define ADD_OPTION_ENUM(PATH, ENUM_HINT, VALUE) \
	r_options->push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::INT, SNAME(PATH), PROPERTY_HINT_ENUM, ENUM_HINT), VALUE));

	ADD_OPTION_ENUM("blender/nodes/visible", "Visible Only,Renderable,All", BLEND_VISIBLE_ALL);
	ADD_OPTION_BOOL("blender/nodes/punctual_lights", true);
	ADD_OPTION_BOOL("blender/nodes/cameras", true);
	ADD_OPTION_BOOL("blender/nodes/custom_properties", true);
	ADD_OPTION_ENUM("blender/nodes/modifiers", "No Modifiers,All Modifiers", BLEND_MODIFIERS_ALL);
	ADD_OPTION_BOOL("blender/meshes/colors", false);
	ADD_OPTION_BOOL("blender/meshes/uvs", true);
	ADD_OPTION_BOOL("blender/meshes/normals", true);
	ADD_OPTION_BOOL("blender/meshes/tangents", true);
	ADD_OPTION_ENUM("blender/meshes/skins", "None,4 Influences (Compatible),All Influences", BLEND_BONE_INFLUENCES_ALL);
	ADD_OPTION_BOOL("blender/meshes/export_bones_deforming_mesh_only", false);
	ADD_OPTION_BOOL("blender/materials/unpack_enabled", true);
	ADD_OPTION_ENUM("blender/materials/export_materials", "Placeholder,Export", BLEND_MATERIAL_EXPORT_EXPORT);
	ADD_OPTION_BOOL("blender/animation/limit_playback", true);
	ADD_OPTION_BOOL("blender/animation/always_sample", true);
	ADD_OPTION_BOOL("blender/animation/group_tracks", true);

#undef ADD_OPTION_BOOL
#undef ADD_OPTION_ENUM
}

#endif // TOOLS_ENABLED
