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

#include "core/error_macros.h"
#include "core/os/file_access.h"
#include "core/project_settings.h"
#include "editor_translation_parser.h"
#include "plugins/packed_scene_translation_parser_plugin.h"

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

	// Collect all translatable strings according to files order in "POT Generation" setting.
	for (int i = 0; i < files.size(); i++) {
		Vector<String> translation_strings;
		String file_path = files[i];
		String file_extension = file_path.get_extension();

		if (EditorTranslationParser::get_singleton()->can_parse(file_extension)) {
			EditorTranslationParser::get_singleton()->get_parser(file_extension)->parse_file(file_path, &translation_strings);
		} else {
			ERR_PRINT("Unrecognized file extension " + file_extension + " in generate_pot()");
			return;
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

		// Split \\n and \n.
		Vector<String> temp = msg.split("\\n");
		Vector<String> msg_lines;
		for (int i = 0; i < temp.size(); i++) {
			msg_lines.append_array(temp[i].split("\n"));
			if (i < temp.size() - 1) {
				// Add \n.
				msg_lines.set(msg_lines.size() - 1, msg_lines[msg_lines.size() - 1] + "\\n");
			}
		}

		// Write msgid.
		file->store_string("msgid ");
		for (int i = 0; i < msg_lines.size(); i++) {
			file->store_line("\"" + msg_lines[i] + "\"");
		}

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
}

POTGenerator::~POTGenerator() {
	memdelete(singleton);
	singleton = nullptr;
}
