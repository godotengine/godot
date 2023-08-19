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

#ifdef TOOLS_ENABLED

#include "../gltf_defines.h"
#include "../gltf_document.h"

uint32_t EditorSceneFormatImporterGLTF::get_import_flags() const {
	return ImportFlags::IMPORT_SCENE | ImportFlags::IMPORT_ANIMATION;
}

void EditorSceneFormatImporterGLTF::get_extensions(List<String> *r_extensions) const {
	r_extensions->push_back("gltf");
	r_extensions->push_back("glb");
}

Node *EditorSceneFormatImporterGLTF::import_scene(const String &p_path, uint32_t p_flags,
		const HashMap<StringName, Variant> &p_options,
		List<String> *r_missing_deps, Error *r_err) {
	Ref<GLTFDocument> gltf;
	gltf.instantiate();
	Ref<GLTFState> state;
	state.instantiate();
	if (p_options.has("gltf/embedded_image_handling")) {
		int32_t enum_option = p_options["gltf/embedded_image_handling"];
		state->set_handle_binary_image(enum_option);
	}
	Error err = gltf->append_from_file(p_path, state, p_flags);
	if (err != OK) {
		if (r_err) {
			*r_err = err;
		}
		return nullptr;
	}
	if (p_options.has("animation/import")) {
		state->set_create_animations(bool(p_options["animation/import"]));
	}

#ifndef DISABLE_DEPRECATED
	bool trimming = p_options.has("animation/trimming") ? (bool)p_options["animation/trimming"] : false;
	bool remove_immutable = p_options.has("animation/remove_immutable_tracks") ? (bool)p_options["animation/remove_immutable_tracks"] : true;
	return gltf->generate_scene(state, (float)p_options["animation/fps"], trimming, remove_immutable);
#else
	return gltf->generate_scene(state, (float)p_options["animation/fps"], (bool)p_options["animation/trimming"], (bool)p_options["animation/remove_immutable_tracks"]);
#endif
}

void EditorSceneFormatImporterGLTF::get_import_options(const String &p_path,
		List<ResourceImporter::ImportOption> *r_options) {
	r_options->push_back(ResourceImporterScene::ImportOption(PropertyInfo(Variant::INT, "gltf/embedded_image_handling", PROPERTY_HINT_ENUM, "Discard All Textures,Extract Textures,Embed As Basis Universal,Embed as Uncompressed", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), GLTFState::HANDLE_BINARY_EXTRACT_TEXTURES));
}

#endif // TOOLS_ENABLED
