/*************************************************************************/
/*  packed_scene_translation_parser_plugin.cpp                           */
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

#include "packed_scene_translation_parser_plugin.h"

#include "core/io/resource_loader.h"
#include "scene/resources/packed_scene.h"

void PackedSceneEditorTranslationParserPlugin::get_recognized_extensions(List<String> *r_extensions) const {
	ResourceLoader::get_recognized_extensions_for_type("PackedScene", r_extensions);
}

Error PackedSceneEditorTranslationParserPlugin::parse_file(const String &p_path, Vector<String> *r_ids, Vector<Vector<String>> *r_ids_ctx_plural) {
	// Parse specific scene Node's properties (see in constructor) that are auto-translated by the engine when set. E.g Label's text property.
	// These properties are translated with the tr() function in the C++ code when being set or updated.

	Error err;
	RES loaded_res = ResourceLoader::load(p_path, "PackedScene", false, &err);
	if (err) {
		ERR_PRINT("Failed to load " + p_path);
		return err;
	}
	Ref<SceneState> state = Ref<PackedScene>(loaded_res)->get_state();

	Vector<String> parsed_strings;
	String property_name;
	Variant property_value;
	for (int i = 0; i < state->get_node_count(); i++) {
		if (!ClassDB::is_parent_class(state->get_node_type(i), "Control") && !ClassDB::is_parent_class(state->get_node_type(i), "Viewport")) {
			continue;
		}

		for (int j = 0; j < state->get_node_property_count(i); j++) {
			property_name = state->get_node_property_name(i, j);
			if (!lookup_properties.has(property_name)) {
				continue;
			}

			property_value = state->get_node_property_value(i, j);

			if (property_name == "script" && property_value.get_type() == Variant::OBJECT && !property_value.is_null()) {
				// Parse built-in script.
				Ref<Script> s = Object::cast_to<Script>(property_value);
				String extension = s->get_language()->get_extension();
				if (EditorTranslationParser::get_singleton()->can_parse(extension)) {
					Vector<String> temp;
					Vector<Vector<String>> ids_context_plural;
					EditorTranslationParser::get_singleton()->get_parser(extension)->parse_file(s->get_path(), &temp, &ids_context_plural);
					parsed_strings.append_array(temp);
					r_ids_ctx_plural->append_array(ids_context_plural);
				}
			} else if (property_name == "filters") {
				// Extract FileDialog's filters property with values in format "*.png ; PNG Images","*.gd ; GDScript Files".
				Vector<String> str_values = property_value;
				for (int k = 0; k < str_values.size(); k++) {
					String desc = str_values[k].get_slice(";", 1).strip_edges();
					if (!desc.empty()) {
						parsed_strings.push_back(desc);
					}
				}
			} else if (property_value.get_type() == Variant::STRING) {
				String str_value = String(property_value);
				// Prevent reading text containing only spaces.
				if (!str_value.strip_edges().empty()) {
					parsed_strings.push_back(str_value);
				}
			}
		}
	}

	r_ids->append_array(parsed_strings);

	return OK;
}

PackedSceneEditorTranslationParserPlugin::PackedSceneEditorTranslationParserPlugin() {
	// Scene Node's properties containing strings that will be fetched for translation.
	lookup_properties.insert("text");
	lookup_properties.insert("hint_tooltip");
	lookup_properties.insert("placeholder_text");
	lookup_properties.insert("dialog_text");
	lookup_properties.insert("filters");
	lookup_properties.insert("script");

	//Add exception list (to prevent false positives)
	//line edit, text edit, richtextlabel
	//Set<String> exception_list;
	//exception_list.insert("RichTextLabel");
}
