/*************************************************************************/
/*  gdscript_translation_parser_plugin.cpp                               */
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

#include "gdscript_translation_parser_plugin.h"

#include "core/io/resource_loader.h"
#include "modules/gdscript/gdscript.h"

void GDScriptEditorTranslationParserPlugin::get_recognized_extensions(List<String> *r_extensions) const {
	GDScriptLanguage::get_singleton()->get_recognized_extensions(r_extensions);
}

Error GDScriptEditorTranslationParserPlugin::parse_file(const String &p_path, Vector<String> *r_extracted_strings) {
	// Parse and match all GDScript function API that involves translation string.
	// E.g get_node("Label").text = "something", var test = tr("something"), "something" will be matched and collected.

	Error err;
	RES loaded_res = ResourceLoader::load(p_path, "", false, &err);
	if (err) {
		ERR_PRINT("Failed to load " + p_path);
		return err;
	}

	Ref<GDScript> gdscript = loaded_res;
	String source_code = gdscript->get_source_code();
	Vector<String> parsed_strings;

	// Search translation strings with RegEx.
	regex.clear();
	regex.compile(String("|").join(patterns));
	Array results = regex.search_all(source_code);
	_get_captured_strings(results, &parsed_strings);

	// Special handling for FileDialog.
	Vector<String> temp;
	_parse_file_dialog(source_code, &temp);
	parsed_strings.append_array(temp);

	// Filter out / and +
	String filter = "(?:\\\\\\n|\"[\\s\\\\]*\\+\\s*\")";
	regex.clear();
	regex.compile(filter);
	for (int i = 0; i < parsed_strings.size(); i++) {
		parsed_strings.set(i, regex.sub(parsed_strings[i], "", true));
	}

	r_extracted_strings->append_array(parsed_strings);

	return OK;
}

void GDScriptEditorTranslationParserPlugin::_parse_file_dialog(const String &p_source_code, Vector<String> *r_output) {
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

void GDScriptEditorTranslationParserPlugin::_get_captured_strings(const Array &p_results, Vector<String> *r_output) {
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

GDScriptEditorTranslationParserPlugin::GDScriptEditorTranslationParserPlugin() {
	// Regex search pattern templates.
	// The extra complication in the regex pattern is to ensure that the matching works when users write over multiple lines, use tabs etc.
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
