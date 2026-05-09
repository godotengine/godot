/**************************************************************************/
/*  editor_scene_importer_gltf.cpp                                        */
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
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "editor_scene_importer_gltf.h"

#include "../gltf_defines.h"
#include "../gltf_document.h"

#include "core/io/resource_importer.h"

#include "scene/resources/mesh.h"

void EditorSceneFormatImporterGLTF::get_extensions(List<String> *r_extensions) const {
	r_extensions->push_back("gltf");
	r_extensions->push_back("glb");
}

Node *EditorSceneFormatImporterGLTF::import_scene(const String &p_path, uint32_t p_flags,
		const HashMap<StringName, Variant> &p_options,
		List<String> *r_missing_deps, Error *r_err) {
	Ref<GLTFDocument> gltf_doc;
	gltf_doc.instantiate();
	Ref<GLTFState> gltf_state;
	gltf_state.instantiate();
	if (p_options.has("gltf/naming_version")) {
		int naming_version = p_options["gltf/naming_version"];
		gltf_doc->set_naming_version(naming_version);
	}
	if (p_options.has("gltf/embedded_image_handling")) {
		int32_t enum_option = p_options["gltf/embedded_image_handling"];
		gltf_state->set_handle_binary_image_mode((GLTFState::HandleBinaryImageMode)enum_option);
	}
	if (p_options.has("gltf/texture_map_mode")) {
		int32_t enum_option = p_options["gltf/texture_map_mode"];
		gltf_doc->set_texture_map_mode((GLTFDocument::TextureMapMode)enum_option);
	}
	if (p_options.has(SNAME("nodes/import_as_skeleton_bones")) ? (bool)p_options[SNAME("nodes/import_as_skeleton_bones")] : false) {
		gltf_state->set_import_as_skeleton_bones(true);
	}
	if (p_options.has(SNAME("extract_path"))) {
		gltf_state->set_extract_path(p_options["extract_path"]);
	}
	gltf_state->set_bake_fps(p_options["animation/fps"]);
	if (p_options.has("physics/is_rigid")) {
		gltf_state->set_import_as_rigid((bool)p_options["physics/is_rigid"]);
	}
	if (p_options.has("physics/jointed")) {
		gltf_state->set_jointed((bool)p_options["physics/jointed"]);
	}
	if (p_options.has("physics/neighboring_distance")) {
		gltf_state->set_neighboring_distance((float)p_options["physics/neighboring_distance"]);
	}
	if (gltf_state->get_import_as_rigid()) {
		const int64_t max_convex_hulls = p_options.has("physics/max_convex_hulls") ? (int64_t)p_options["physics/max_convex_hulls"] : (int64_t)8;
		const double max_concavity = p_options.has("physics/max_concavity") ? (double)p_options["physics/max_concavity"] : 0.5;
		const int64_t resolution = p_options.has("physics/resolution") ? (int64_t)p_options["physics/resolution"] : (int64_t)10000;
		const double min_volume_per_convex_hull = p_options.has("physics/min_volume_per_convex_hull") ? (double)p_options["physics/min_volume_per_convex_hull"] : 0.0001;

		Ref<MeshConvexDecompositionSettings> settings;
		settings.instantiate();
		settings->set_max_convex_hulls((uint32_t)MAX((int64_t)1, max_convex_hulls));
		settings->set_max_concavity((real_t)max_concavity);
		settings->set_resolution((uint32_t)MAX((int64_t)1, resolution));
		settings->set_min_volume_per_convex_hull((real_t)min_volume_per_convex_hull);
		gltf_state->set_convex_decomposition_settings(settings);
	}
	Error err = gltf_doc->append_from_file(p_path, gltf_state, p_flags);
	if (err != OK) {
		if (r_err) {
			*r_err = err;
		}
		return nullptr;
	}
	if (p_options.has("animation/import")) {
		gltf_state->set_create_animations(bool(p_options["animation/import"]));
	}

#ifndef DISABLE_DEPRECATED
	bool trimming = p_options.has("animation/trimming") ? (bool)p_options["animation/trimming"] : false;
	return gltf_doc->generate_scene(gltf_state, gltf_state->get_bake_fps(), trimming, false);
#else
	return gltf_doc->generate_scene(gltf_state, gltf_state->get_bake_fps(), (bool)p_options["animation/trimming"], false);
#endif
}

void EditorSceneFormatImporterGLTF::get_import_options(const String &p_path,
		List<ResourceImporter::ImportOption> *r_options) {
	String file_extension = p_path.get_extension().to_lower();
	// Returns all the options when path is empty because that means it's for the Project Settings.
	if (p_path.is_empty() || file_extension == "gltf" || file_extension == "glb") {
		r_options->push_back(ResourceImporterScene::ImportOption(PropertyInfo(Variant::INT, "gltf/naming_version", PROPERTY_HINT_ENUM, "Godot 4.0 or 4.1,Godot 4.2 to 4.4,Godot 4.5 or later"), 2));
		r_options->push_back(ResourceImporterScene::ImportOption(PropertyInfo(Variant::INT, "gltf/embedded_image_handling", PROPERTY_HINT_ENUM, "Discard All Textures,Extract Textures,Embed as Basis Universal,Embed as Uncompressed", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), GLTFState::HANDLE_BINARY_IMAGE_MODE_EXTRACT_TEXTURES));
		r_options->push_back(ResourceImporterScene::ImportOption(PropertyInfo(Variant::INT, "gltf/texture_map_mode", PROPERTY_HINT_ENUM, "Do Not Remap,Remap to StandardMaterial3D", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), GLTFDocument::TEXTURE_MAP_MODE_REMAP_TO_STANDARD_MATERIAL));

		r_options->push_back(ResourceImporterScene::ImportOption(PropertyInfo(Variant::BOOL, "physics/is_rigid", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
		r_options->push_back(ResourceImporterScene::ImportOption(PropertyInfo(Variant::BOOL, "physics/jointed", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
		r_options->push_back(ResourceImporterScene::ImportOption(PropertyInfo(Variant::FLOAT, "physics/neighboring_distance", PROPERTY_HINT_RANGE, "0,100000,0.0001,or_greater", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 1.0));
		r_options->push_back(ResourceImporterScene::ImportOption(PropertyInfo(Variant::INT, "physics/max_convex_hulls", PROPERTY_HINT_RANGE, "1,64,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 8));
		r_options->push_back(ResourceImporterScene::ImportOption(PropertyInfo(Variant::FLOAT, "physics/max_concavity", PROPERTY_HINT_RANGE, "0,1,0.001", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 0.5));
		r_options->push_back(ResourceImporterScene::ImportOption(PropertyInfo(Variant::INT, "physics/resolution", PROPERTY_HINT_RANGE, "1,100000,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 10000));
		r_options->push_back(ResourceImporterScene::ImportOption(PropertyInfo(Variant::FLOAT, "physics/min_volume_per_convex_hull", PROPERTY_HINT_RANGE, "0,1,0.000001,or_greater", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 0.0001));
	}
}

void EditorSceneFormatImporterGLTF::handle_compatibility_options(HashMap<StringName, Variant> &p_import_params) const {
	if (!p_import_params.has("gltf/texture_map_mode")) {
		// If an existing import file is missing the glTF
		// texture map mode, we need to use "Do Not Remap".
		p_import_params["gltf/texture_map_mode"] = (int64_t)GLTFDocument::TEXTURE_MAP_MODE_DO_NOT_REMAP;
	}
}

Variant EditorSceneFormatImporterGLTF::get_option_visibility(const String &p_path, const String &p_scene_import_type,
		const String &p_option, const HashMap<StringName, Variant> &p_options) {
	return true;
}
