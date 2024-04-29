/**************************************************************************/
/*  packed_scene_translation_parser_plugin.cpp                            */
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

#include "packed_scene_translation_parser_plugin.h"

#include "core/io/resource_loader.h"
#include "core/object/script_language.h"
#include "scene/gui/option_button.h"
#include "scene/resources/packed_scene.h"

void PackedSceneEditorTranslationParserPlugin::get_recognized_extensions(List<String> *r_extensions) const {
	ResourceLoader::get_recognized_extensions_for_type("PackedScene", r_extensions);
}

Error PackedSceneEditorTranslationParserPlugin::parse_file(const String &p_path, Vector<String> *r_ids, Vector<Vector<String>> *r_ids_ctx_plural) {
	// Parse specific scene Node's properties (see in constructor) that are auto-translated by the engine when set. E.g Label's text property.
	// These properties are translated with the tr() function in the C++ code when being set or updated.

	Error err;
	Ref<Resource> loaded_res = ResourceLoader::load(p_path, "PackedScene", ResourceFormatLoader::CACHE_MODE_REUSE, &err);
	if (err) {
		ERR_PRINT("Failed to load " + p_path);
		return err;
	}
	Ref<SceneState> state = Ref<PackedScene>(loaded_res)->get_state();

	Vector<String> parsed_strings;
	for (int i = 0; i < state->get_node_count(); i++) {
		String node_type = state->get_node_type(i);
		if (!ClassDB::is_parent_class(node_type, "Control") && !ClassDB::is_parent_class(node_type, "Window")) {
			continue;
		}

		// Find the `auto_translate` property, and abort the string parsing of the node if disabled.
		bool auto_translating = true;
		for (int j = 0; j < state->get_node_property_count(i); j++) {
			if (state->get_node_property_name(i, j) == "auto_translate" && (bool)state->get_node_property_value(i, j) == false) {
				auto_translating = false;
				break;
			}
		}
		if (!auto_translating) {
			continue;
		}

		for (int j = 0; j < state->get_node_property_count(i); j++) {
			String property_name = state->get_node_property_name(i, j);
			if (!lookup_properties.has(property_name) || (exception_list.has(node_type) && exception_list[node_type].has(property_name))) {
				continue;
			}

			Variant property_value = state->get_node_property_value(i, j);
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
			} else if ((node_type == "MenuButton" || node_type == "OptionButton") && property_name == "items") {
				Vector<String> str_values = property_value;
				int incr_value = node_type == "MenuButton" ? PopupMenu::ITEM_PROPERTY_SIZE : OptionButton::ITEM_PROPERTY_SIZE;
				for (int k = 0; k < str_values.size(); k += incr_value) {
					String desc = str_values[k].get_slice(";", 1).strip_edges();
					if (!desc.is_empty()) {
						parsed_strings.push_back(desc);
					}
				}
			} else if (node_type == "FileDialog" && property_name == "filters") {
				// Extract FileDialog's filters property with values in format "*.png ; PNG Images","*.gd ; GDScript Files".
				Vector<String> str_values = property_value;
				for (int k = 0; k < str_values.size(); k++) {
					String desc = str_values[k].get_slice(";", 1).strip_edges();
					if (!desc.is_empty()) {
						parsed_strings.push_back(desc);
					}
				}
			} else if (property_value.get_type() == Variant::STRING) {
				String str_value = String(property_value);
				// Prevent reading text containing only spaces.
				if (!str_value.strip_edges().is_empty()) {
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
	lookup_properties.insert("tooltip_text");
	lookup_properties.insert("placeholder_text");
	lookup_properties.insert("items");
	lookup_properties.insert("title");
	lookup_properties.insert("dialog_text");
	lookup_properties.insert("filters");
	lookup_properties.insert("script");

	// Exception list (to prevent false positives).
	exception_list.insert("LineEdit", { "text" });
	exception_list.insert("TextEdit", { "text" });
	exception_list.insert("CodeEdit", { "text" });
}
