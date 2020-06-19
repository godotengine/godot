/*************************************************************************/
/*  pot_generator.cpp                                                    */
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

#include "pot_generator.h"

#include "core/class_db.h"
#include "core/error_macros.h"
#include "core/io/resource_loader.h"
#include "core/project_settings.h"
#include "core/script_language.h"
#include "core/variant.h"
#include "modules/gdscript/gdscript.h"
#include "scene/resources/packed_scene.h"

POTGenerator *POTGenerator::singleton = nullptr;

//#define DEBUG_POT

#ifdef DEBUG_POT
void _print_all_translation_strings(const OrderedHashMap<String, Set<String>> &p_all_translation_strings) {
	for (auto E_pair = p_all_translation_strings.front(); E_pair; E_pair = E_pair.next()) {
		String msg = static_cast<String>(E_pair.key()) + " : ";
		for (Set<String>::Element *E = E_pair.value().front(); E; E = E->next()) {
			msg += E->get() + " ";
		}
		print_line(msg);
	}
}
#endif

void POTGenerator::generate_pot(const String &p_file) {
	if (!ProjectSettings::get_singleton()->has_setting("locale/translations_pot_files")) {
		WARN_PRINT("No files selected for POT generation.");
		return;
	}

	// Clear all_translation_strings of the previous round.
	all_translation_strings.clear();

	Vector<String> files = ProjectSettings::get_singleton()->get("locale/translations_pot_files");
	Vector<String> translation_strings;
	List<String> packed_scene_extensions;
	ResourceLoader::get_recognized_extensions_for_type("PackedScene", &packed_scene_extensions);

	// Collect all translatable strings according to files order in "POT Generation" setting.
	for (int i = 0; i < files.size(); i++) {
		String file_path = files[i];
		String file_extension = file_path.get_extension();

		Error err;
		RES loaded_res = ResourceLoader::load(file_path, "", false, &err);
		if (err) {
			ERR_PRINT("Failed to load " + file_path);
			continue;
		}

		bool is_scene_file = false;
		for (List<String>::Element *E = packed_scene_extensions.front(); E; E = E->next()) {
			if (file_extension == E->get()) {
				is_scene_file = true;
				break;
			}
		}

		if (is_scene_file) {
			Ref<PackedScene> ps = loaded_res;
			translation_strings = _parse_scene(ps->get_state());
		} else if (file_extension == "gd") {
			// Currently we only support GDScript.
			Ref<GDScript> s = loaded_res;
			translation_strings = _parse_script(s->get_source_code());
		} else {
			ERR_PRINT("Unrecognized file extension in generate_pot()");
			continue;
		}

		// Store translation strings parsed in this iteration along with their corresponding source file - to write into POT later on.
		for (int j = 0; j < translation_strings.size(); j++) {
			all_translation_strings[translation_strings[j]].insert(file_path);
		}
	}

#ifdef DEBUG_POT
	_print_all_translation_strings(all_translation_strings);
#endif

	_write_to_pot(p_file);
}

Vector<String> POTGenerator::_parse_scene(const Ref<SceneState> &p_state) {
	// Parse specific scene Node's properties (see in constructor) that are auto-translated by the engine when set. E.g Label's text property.
	// These properties are translated with the tr() function in the C++ code when being set or updated.

	Vector<String> parsed_strings;

	String property_name;
	Variant property_value;
	for (int i = 0; i < p_state->get_node_count(); i++) {
		if (!ClassDB::is_parent_class(p_state->get_node_type(i), "Control") && !ClassDB::is_parent_class(p_state->get_node_type(i), "Viewport")) {
			continue;
		}

		for (int j = 0; j < p_state->get_node_property_count(i); j++) {
			property_name = p_state->get_node_property_name(i, j);
			if (!lookup_properties.has(property_name)) {
				continue;
			}

			property_value = p_state->get_node_property_value(i, j);

			if (property_name == "script" && property_value.get_type() == Variant::OBJECT && !property_value.is_null()) {
				// Parse built-in script. Currently we only support GDScript.
				Ref<GDScript> s = Object::cast_to<GDScript>(property_value);
				parsed_strings.append_array(_parse_script(s->get_source_code()));
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

	return parsed_strings;
}

Vector<String> POTGenerator::_parse_script(const String &p_source_code) {
	// Parse and match all GDScript function API that involves translation string.
	// E.g get_node("Label").text = "something", var test = tr("something")
	// "something" will be matched and collected
	// The extra complication in the regex pattern is to ensure that the matching works when users write over multiple lines, use tabs etc.

	Vector<String> parsed_strings;

	regex.clear();
	regex.compile(String("|").join(patterns));
	Array results = regex.search_all(p_source_code);
	_get_captured_strings(results, &parsed_strings);

	// Special handling for FileDialog
	Vector<String> temp;
	_parse_file_dialog(p_source_code, &temp);
	parsed_strings.append_array(temp);

	// Filter out / and +
	String filter = "(?:\\\\\\n|\"[\\s\\\\]*\\+\\s*\")";
	regex.clear();
	regex.compile(filter);
	for (int i = 0; i < parsed_strings.size(); i++) {
		parsed_strings.set(i, regex.sub(parsed_strings[i], "", true));
	}

	return parsed_strings;
}

void POTGenerator::_parse_file_dialog(const String &p_source_code, Vector<String> *r_output) {
	// FileDialog API has the form .filters = PackedStringArray(["*.png ; PNG Images","*.gd ; GDScript Files"]).
	// First filter: Get "*.png ; PNG Images", "*.gd ; GDScript Files" from PackedStringArray.
	regex.clear();
	regex.compile(String("|").join(file_dialog_patterns));
	Array results = regex.search_all(p_source_code);

	Vector<String> temp;
	_get_captured_strings(results, &temp);
	String captured_strings = String(",").join(temp);

	// Second filter: Get the texts after semicolon from "*.png ; PNG Images","*.gd ; GDScript Files".
	String second_filter = "\"[^;]+;" + text + "\"";
	regex.clear();
	regex.compile(second_filter);
	results = regex.search_all(captured_strings);
	_get_captured_strings(results, r_output);
	for (int i = 0; i < r_output->size(); i++) {
		r_output->set(i, r_output->get(i).strip_edges());
	}
}

void POTGenerator::_get_captured_strings(const Array &p_results, Vector<String> *r_output) {
	Ref<RegExMatch> result;
	for (int i = 0; i < p_results.size(); i++) {
		result = p_results[i];
		for (int j = 0; j < result->get_group_count(); j++) {
			String s = result->get_string(j + 1);
			// Prevent reading text with only spaces.
			if (!s.strip_edges().empty()) {
				r_output->push_back(s);
			}
		}
	}
}

void POTGenerator::_write_to_pot(const String &p_file) {
	Error err;
	FileAccess *file = FileAccess::open(p_file, FileAccess::WRITE, &err);
	if (err != OK) {
		ERR_PRINT("Failed to open " + p_file);
		return;
	}

	String project_name = ProjectSettings::get_singleton()->get("application/config/name");
	Vector<String> files = ProjectSettings::get_singleton()->get("locale/translations_pot_files");
	String extracted_files = "";
	for (int i = 0; i < files.size(); i++) {
		extracted_files += "# " + files[i] + "\n";
	}
	const String header =
			"# LANGUAGE translation for " + project_name + " for the following files:\n" + extracted_files +
			"#\n"
			"#\n"
			"# FIRST AUTHOR < EMAIL @ADDRESS>, YEAR.\n"
			"#\n"
			"#, fuzzy\n"
			"msgid \"\"\n"
			"msgstr \"\"\n"
			"\"Project-Id-Version: " +
			project_name + "\\n\"\n"
						   "\"Content-Type: text/plain; charset=UTF-8\\n\"\n"
						   "\"Content-Transfer-Encoding: 8-bit\\n\"\n\n";

	file->store_string(header);

	for (OrderedHashMap<String, Set<String>>::Element E_pair = all_translation_strings.front(); E_pair; E_pair = E_pair.next()) {
		String msg = E_pair.key();

		// Write file locations.
		for (Set<String>::Element *E = E_pair.value().front(); E; E = E->next()) {
			file->store_line("#: " + E->get().trim_prefix("res://"));
		}

		// Write msgid.
		Vector<String> msg_lines = msg.split("\\n");
		file->store_string("msgid \"" + msg_lines[0]);
		for (int i = 1; i < msg_lines.size(); i++) {
			file->store_line("\\n\"");
			file->store_string("\"" + msg_lines[i]);
		}
		file->store_line("\"");

		file->store_line("msgstr \"\"\n");
	}

	file->close();
}

POTGenerator *POTGenerator::get_singleton() {
	if (!singleton) {
		singleton = memnew(POTGenerator);
	}
	return singleton;
}

POTGenerator::POTGenerator() {
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

	// Regex search pattern templates.
	const String dot = "\\.[\\s\\\\]*";
	const String str_assign_template = "[\\s\\\\]*=[\\s\\\\]*\"" + text + "\"";
	const String first_arg_template = "[\\s\\\\]*\\([\\s\\\\]*\"" + text + "\"[\\s\\S]*?\\)";
	const String second_arg_template = "[\\s\\\\]*\\([\\s\\S]+?,[\\s\\\\]*\"" + text + "\"[\\s\\S]*?\\)";

	// Common patterns.
	patterns.push_back("tr" + first_arg_template);
	patterns.push_back(dot + "text" + str_assign_template);
	patterns.push_back(dot + "placeholder_text" + str_assign_template);
	patterns.push_back(dot + "hint_tooltip" + str_assign_template);
	patterns.push_back(dot + "set_text" + first_arg_template);
	patterns.push_back(dot + "set_tooltip" + first_arg_template);
	patterns.push_back(dot + "set_placeholder" + first_arg_template);

	// Tabs and TabContainer API.
	patterns.push_back(dot + "set_tab_title" + second_arg_template);
	patterns.push_back(dot + "add_tab" + first_arg_template);

	// PopupMenu API.
	patterns.push_back(dot + "add_check_item" + first_arg_template);
	patterns.push_back(dot + "add_icon_check_item" + second_arg_template);
	patterns.push_back(dot + "add_icon_item" + second_arg_template);
	patterns.push_back(dot + "add_icon_radio_check_item" + second_arg_template);
	patterns.push_back(dot + "add_item" + first_arg_template);
	patterns.push_back(dot + "add_multistate_item" + first_arg_template);
	patterns.push_back(dot + "add_radio_check_item" + first_arg_template);
	patterns.push_back(dot + "add_separator" + first_arg_template);
	patterns.push_back(dot + "add_submenu_item" + first_arg_template);
	patterns.push_back(dot + "set_item_text" + second_arg_template);
	//patterns.push_back(dot + "set_item_tooltip" + second_arg_template); //no tr() behind this function. might be bug.

	// FileDialog API - special case.
	const String fd_text = "((?:[\\s\\\\]*\"(?:[^\"\\\\]|\\\\[\\s\\S])*(?:\"[\\s\\\\]*\\+[\\s\\\\]*\"(?:[^\"\\\\]|\\\\[\\s\\S])*)*\"[\\s\\\\]*,?)*)";
	const String packed_string_array = "[\\s\\\\]*PackedStringArray[\\s\\\\]*\\([\\s\\\\]*\\[" + fd_text + "\\][\\s\\\\]*\\)";
	file_dialog_patterns.push_back(dot + "add_filter[\\s\\\\]*\\(" + fd_text + "[\\s\\\\]*\\)");
	file_dialog_patterns.push_back(dot + "filters[\\s\\\\]*=" + packed_string_array);
	file_dialog_patterns.push_back(dot + "set_filters[\\s\\\\]*\\(" + packed_string_array + "[\\s\\\\]*\\)");
}

POTGenerator::~POTGenerator() {
	memdelete(singleton);
	singleton = nullptr;
}
