/*************************************************************************/
/*  pot_generator.cpp                                                    */
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

#include "pot_generator.h"

#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "editor_translation_parser.h"
#include "plugins/packed_scene_translation_parser_plugin.h"

POTGenerator *POTGenerator::singleton = nullptr;

#ifdef DEBUG_POT
void POTGenerator::_print_all_translation_strings() {
	for (OrderedHashMap<String, Vector<POTGenerator::MsgidData>>::Element E = all_translation_strings.front(); E; E = E.next()) {
		Vector<MsgidData> v_md = all_translation_strings[E.key()];
		for (int i = 0; i < v_md.size(); i++) {
			print_line("++++++");
			print_line("msgid: " + E.key());
			print_line("context: " + v_md[i].ctx);
			print_line("msgid_plural: " + v_md[i].plural);
			for (Set<String>::Element *E = v_md[i].locations.front(); E; E = E->next()) {
				print_line("location: " + E->get());
			}
		}
	}
}
#endif

void POTGenerator::generate_pot(const String &p_file) {
	if (!ProjectSettings::get_singleton()->has_setting("internationalization/locale/translations_pot_files")) {
		WARN_PRINT("No files selected for POT generation.");
		return;
	}

	// Clear all_translation_strings of the previous round.
	all_translation_strings.clear();

	Vector<String> files = ProjectSettings::get_singleton()->get("internationalization/locale/translations_pot_files");

	// Collect all translatable strings according to files order in "POT Generation" setting.
	for (int i = 0; i < files.size(); i++) {
		Vector<String> msgids;
		Vector<Vector<String>> msgids_context_plural;
		String file_path = files[i];
		String file_extension = file_path.get_extension();

		if (EditorTranslationParser::get_singleton()->can_parse(file_extension)) {
			EditorTranslationParser::get_singleton()->get_parser(file_extension)->parse_file(file_path, &msgids, &msgids_context_plural);
		} else {
			ERR_PRINT("Unrecognized file extension " + file_extension + " in generate_pot()");
			return;
		}

		for (int j = 0; j < msgids_context_plural.size(); j++) {
			Vector<String> entry = msgids_context_plural[j];
			_add_new_msgid(entry[0], entry[1], entry[2], file_path);
		}
		for (int j = 0; j < msgids.size(); j++) {
			_add_new_msgid(msgids[j], "", "", file_path);
		}
	}

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
	Vector<String> files = ProjectSettings::get_singleton()->get("internationalization/locale/translations_pot_files");
	String extracted_files = "";
	for (int i = 0; i < files.size(); i++) {
		extracted_files += "# " + files[i] + "\n";
	}
	const String header =
			"# LANGUAGE translation for " + project_name + " for the following files:\n" + extracted_files +
			"#\n"
			"# FIRST AUTHOR < EMAIL @ADDRESS>, YEAR.\n"
			"#\n"
			"#, fuzzy\n"
			"msgid \"\"\n"
			"msgstr \"\"\n"
			"\"Project-Id-Version: " +
			project_name + "\\n\"\n"
						   "\"MIME-Version: 1.0\\n\"\n"
						   "\"Content-Type: text/plain; charset=UTF-8\\n\"\n"
						   "\"Content-Transfer-Encoding: 8-bit\\n\"\n";

	file->store_string(header);

	for (OrderedHashMap<String, Vector<MsgidData>>::Element E_pair = all_translation_strings.front(); E_pair; E_pair = E_pair.next()) {
		String msgid = E_pair.key();
		Vector<MsgidData> v_msgid_data = E_pair.value();
		for (int i = 0; i < v_msgid_data.size(); i++) {
			String context = v_msgid_data[i].ctx;
			String plural = v_msgid_data[i].plural;
			const Set<String> &locations = v_msgid_data[i].locations;

			// Put the blank line at the start, to avoid a double at the end when closing the file.
			file->store_line("");

			// Write file locations.
			for (Set<String>::Element *E = locations.front(); E; E = E->next()) {
				file->store_line("#: " + E->get().trim_prefix("res://"));
			}

			// Write context.
			if (!context.is_empty()) {
				file->store_line("msgctxt \"" + context + "\"");
			}

			// Write msgid.
			_write_msgid(file, msgid, false);

			// Write msgid_plural.
			if (!plural.is_empty()) {
				_write_msgid(file, plural, true);
				file->store_line("msgstr[0] \"\"");
				file->store_line("msgstr[1] \"\"");
			} else {
				file->store_line("msgstr \"\"");
			}
		}
	}

	file->close();
}

void POTGenerator::_write_msgid(FileAccess *r_file, const String &p_id, bool p_plural) {
	// Split \\n and \n.
	Vector<String> temp = p_id.split("\\n");
	Vector<String> msg_lines;
	for (int i = 0; i < temp.size(); i++) {
		msg_lines.append_array(temp[i].split("\n"));
		if (i < temp.size() - 1) {
			// Add \n.
			msg_lines.set(msg_lines.size() - 1, msg_lines[msg_lines.size() - 1] + "\\n");
		}
	}

	if (p_plural) {
		r_file->store_string("msgid_plural ");
	} else {
		r_file->store_string("msgid ");
	}

	for (int i = 0; i < msg_lines.size(); i++) {
		r_file->store_line("\"" + msg_lines[i] + "\"");
	}
}

void POTGenerator::_add_new_msgid(const String &p_msgid, const String &p_context, const String &p_plural, const String &p_location) {
	// Insert new location if msgid under same context exists already.
	if (all_translation_strings.has(p_msgid)) {
		Vector<MsgidData> &v_mdata = all_translation_strings[p_msgid];
		for (int i = 0; i < v_mdata.size(); i++) {
			if (v_mdata[i].ctx == p_context) {
				if (!v_mdata[i].plural.is_empty() && !p_plural.is_empty() && v_mdata[i].plural != p_plural) {
					WARN_PRINT("Redefinition of plural message (msgid_plural), under the same message (msgid) and context (msgctxt)");
				}
				v_mdata.write[i].locations.insert(p_location);
				return;
			}
		}
	}

	// Add a new entry of msgid, context, plural and location - context and plural might be empty if the inserted msgid doesn't associated
	// context or plurals.
	MsgidData mdata;
	mdata.ctx = p_context;
	mdata.plural = p_plural;
	mdata.locations.insert(p_location);
	all_translation_strings[p_msgid].push_back(mdata);
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
