/**************************************************************************/
/*  editor_scene_importer_qbo.cpp                                         */
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

#include "editor_scene_importer_qbo.h"

#ifdef TOOLS_ENABLED

#include "modules/qbo/qbo_document.h"

void EditorSceneFormatImporterQBO::get_extensions(List<String> *r_extensions) const {
	r_extensions->push_back("qbo");
}

Node *EditorSceneFormatImporterQBO::import_scene(const String &p_path, uint32_t p_flags,
		const HashMap<StringName, Variant> &p_options,
		List<String> *r_missing_deps, Error *r_err) {
	Ref<QBODocument> gltf;
	gltf.instantiate();
	Ref<GLTFState> state;
	state.instantiate();
	if (p_options.has("gltf/naming_version")) {
		int naming_version = p_options["gltf/naming_version"];
		gltf->set_naming_version(naming_version);
	}
	if (p_options.has("gltf/embedded_image_handling")) {
		int32_t enum_option = p_options["gltf/embedded_image_handling"];
		state->set_handle_binary_image(enum_option);
	}
	if (p_options.has(SNAME("nodes/import_as_skeleton_bones")) ? (bool)p_options[SNAME("nodes/import_as_skeleton_bones")] : false) {
		state->set_import_as_skeleton_bones(true);
	}
	if (p_options.has(SNAME("extract_path"))) {
		state->set_extract_path(p_options["extract_path"]);
	}
	state->set_bake_fps(p_options["animation/fps"]);
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

	return gltf->generate_scene(state, state->get_bake_fps(), (bool)p_options["animation/trimming"], false);
}

#endif // TOOLS_ENABLED
