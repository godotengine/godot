/**************************************************************************/
/*  editor_scene_importer_usd.cpp                                        */
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

#include "editor_scene_importer_usd.h"

#include "../structures/usd_document.h"
#include "../structures/usd_state.h"

#include "core/config/project_settings.h"

void EditorSceneFormatImporterUSD::get_extensions(List<String> *r_extensions) const {
	r_extensions->push_back("usd");
	r_extensions->push_back("usda");
	r_extensions->push_back("usdc");
	r_extensions->push_back("usdz");
}

Node *EditorSceneFormatImporterUSD::import_scene(const String &p_path, uint32_t p_flags,
		const HashMap<StringName, Variant> &p_options,
		List<String> *r_missing_deps, Error *r_err) {
	Ref<USDDocument> doc;
	doc.instantiate();
	Ref<USDState> state;
	state.instantiate();
	print_verbose(vformat("USD path: %s", p_path));
	String path = ProjectSettings::get_singleton()->globalize_path(p_path);
	state->set_bake_fps(p_options.has("animation/fps") ? (double)p_options["animation/fps"] : 30.0);
	Error err = doc->append_from_file(path, state, p_flags, p_path.get_base_dir());
	if (err != OK) {
		if (r_err) {
			*r_err = FAILED;
		}
		return nullptr;
	}
	return doc->generate_scene(state, state->get_bake_fps(), false, false);
}

void EditorSceneFormatImporterUSD::get_import_options(const String &p_path,
		List<ResourceImporter::ImportOption> *r_options) {
	// Returns all the options when path is empty because that means it's for the Project Settings.
	if (p_path.is_empty() || p_path.has_extension("usd") || p_path.has_extension("usda") || p_path.has_extension("usdc") || p_path.has_extension("usdz")) {
		r_options->push_back(ResourceImporterScene::ImportOption(PropertyInfo(Variant::INT, "usd/coordinate_system", PROPERTY_HINT_ENUM, "Auto,Force Y-Up,Force Z-Up"), 0));
	}
}

Variant EditorSceneFormatImporterUSD::get_option_visibility(const String &p_path, const String &p_scene_import_type,
		const String &p_option, const HashMap<StringName, Variant> &p_options) {
	return true;
}
