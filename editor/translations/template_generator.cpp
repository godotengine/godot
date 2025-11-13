/**************************************************************************/
/*  template_generator.cpp                                                */
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

#include "template_generator.h"

#include "core/config/project_settings.h"
#include "editor/translations/editor_translation.h"
#include "editor/translations/editor_translation_parser.h"

Ref<Translation> TranslationTemplateGenerator::parse(const Vector<String> &p_sources, bool p_add_builtin) const {
	Vector<Vector<String>> raw;

	for (const String &path : p_sources) {
		Vector<Vector<String>> parsed_from_file;

		const String &extension = path.get_extension();
		ERR_CONTINUE_MSG(!EditorTranslationParser::get_singleton()->can_parse(extension), vformat("Cannot parse file '%s': unrecognized file extension. Skipping.", path));

		EditorTranslationParser::get_singleton()->get_parser(extension)->parse_file(path, &parsed_from_file);

		for (const Vector<String> &entry : parsed_from_file) {
			ERR_CONTINUE(entry.is_empty());

			const String &msgctxt = (entry.size() > 1) ? entry[1] : String();
			const String &msgid_plural = (entry.size() > 2) ? entry[2] : String();
			const String &comment = (entry.size() > 3) ? entry[3] : String();
			const int source_line = (entry.size() > 4) ? entry[4].to_int() : 0;
			const String &location = source_line > 0 ? vformat("%s:%d", path, source_line) : path;

			raw.push_back({ entry[0], msgctxt, msgid_plural, comment, location });
		}
	}

	if (p_add_builtin) {
		for (const Vector<String> &extractable_msgids : get_extractable_message_list()) {
			raw.push_back({ extractable_msgids[0], extractable_msgids[1], extractable_msgids[2], String(), String() });
		}
	}

	if (GLOBAL_GET("application/config/name_localized").operator Dictionary().is_empty()) {
		const String &project_name = GLOBAL_GET("application/config/name");
		if (!project_name.is_empty()) {
			raw.push_back({ project_name, String(), String(), String(), String() });
		}
	}

	// To remove duplicates.
	HashSet<String> locations;
	HashSet<String> comments;

	Ref<Translation> tpl;
	tpl.instantiate();

	for (const Vector<String> &entry : raw) {
		const String &msgid = entry[0];
		const String &msgctxt = entry[1];
		const String &plural = entry[2];
		const String &comment = entry[3];
		const String &location = entry[4];

		if (tpl->has_message(msgid, msgctxt)) {
			const String &existing_plural = tpl->get_hint(msgid, msgctxt, Translation::HINT_PLURAL);
			if (!existing_plural.is_empty() && !plural.is_empty() && existing_plural != plural) {
				WARN_PRINT(vformat(R"(Skipping different plural definitions for msgid "%s" msgctxt "%s": "%s" and "%s")", msgid, msgctxt, existing_plural, plural));
				continue;
			}
		} else if (plural.is_empty()) {
			tpl->add_message(msgid, StringName(), msgctxt);
		} else {
			tpl->add_plural_message(msgid, { StringName() }, msgctxt);
			tpl->set_hint(msgid, msgctxt, Translation::HINT_PLURAL, plural);
		}

		if (!location.is_empty() && !locations.has(location)) {
			locations.insert(location);

			const String &existing_references = tpl->get_hint(msgid, msgctxt, Translation::HINT_LOCATIONS);
			const String value = existing_references.is_empty() ? location : existing_references + "\n" + location;
			tpl->set_hint(msgid, msgctxt, Translation::HINT_LOCATIONS, value);
		}

		if (!comment.is_empty() && !comments.has(comment)) {
			comments.insert(comment);

			const String &existing_comments = tpl->get_hint(msgid, msgctxt, Translation::HINT_COMMENTS);
			const String value = existing_comments.is_empty() ? comment : existing_comments + "\n" + comment;
			tpl->set_hint(msgid, msgctxt, Translation::HINT_COMMENTS, value);
		}
	}
	return tpl;
}

void TranslationTemplateGenerator::generate(const String &p_file) {
	const Vector<String> files = GLOBAL_GET("internationalization/locale/translations_pot_files");
	const bool add_builtin = GLOBAL_GET("internationalization/locale/translation_add_builtin_strings_to_pot");

	const Ref<Translation> &tpl = parse(files, add_builtin);
	if (tpl->get_message_count()) {
		WARN_PRINT_ED(TTR("No translatable strings found."));
	}

	Error err;
	Ref<FileAccess> file = FileAccess::open(p_file, FileAccess::WRITE, &err);
	ERR_FAIL_COND_MSG(err != OK, "Failed to open " + p_file);

	const String ext = p_file.get_extension().to_lower();
	if (ext == "pot") {
		_write_to_pot(file, tpl);
	} else if (ext == "csv") {
		_write_to_csv(file, tpl);
	} else {
		ERR_FAIL_MSG("Unrecognized translation template file extension: " + ext);
	}
}

static void _write_pot_field(Ref<FileAccess> p_file, const String &p_name, const String &p_value) {
	p_file->store_string(p_name + " ");

	if (p_value.is_empty()) {
		p_file->store_line("\"\"");
		return;
	}

	const Vector<String> lines = p_value.split("\n");
	DEV_ASSERT(lines.size() > 0);

	const String &last_line = lines[lines.size() - 1];
	const int pot_line_count = last_line.is_empty() ? lines.size() - 1 : lines.size();

	if (pot_line_count > 1) {
		p_file->store_line("\"\"");
	}

	for (int i = 0; i < lines.size() - 1; i++) {
		p_file->store_line((lines[i] + "\n").json_escape().quote());
	}
	if (!last_line.is_empty()) {
		p_file->store_line(last_line.json_escape().quote());
	}
}

void TranslationTemplateGenerator::_write_to_pot(Ref<FileAccess> p_file, const Ref<Translation> &p_template) const {
	const String project_name = GLOBAL_GET("application/config/name").operator String().replace("\n", "\\n");
	const Vector<String> files = GLOBAL_GET("internationalization/locale/translations_pot_files");
	String extracted_files;
	for (const String &file : files) {
		extracted_files += "# " + file.replace("\n", "\\n") + "\n";
	}
	const String header =
			"# LANGUAGE translation for " + project_name + " for the following files:\n" +
			extracted_files +
			"#\n"
			"# FIRST AUTHOR <EMAIL@ADDRESS>, YEAR.\n"
			"#\n"
			"#, fuzzy\n"
			"msgid \"\"\n"
			"msgstr \"\"\n"
			"\"Project-Id-Version: " +
			project_name +
			"\\n\"\n"
			"\"MIME-Version: 1.0\\n\"\n"
			"\"Content-Type: text/plain; charset=UTF-8\\n\"\n"
			"\"Content-Transfer-Encoding: 8-bit\\n\"\n";
	p_file->store_string(header);

	List<Translation::MessageKey> keys;
	p_template->get_message_list(&keys);

	for (const Translation::MessageKey &key : keys) {
		// Put the blank line at the start, to avoid a double at the end when closing the file.
		p_file->store_line("");

		// Write comments.
		const Vector<String> comment_lines = p_template->get_hint(key.msgid, key.msgctxt, Translation::HINT_COMMENTS).split("\n");
		for (int i = 0; i < comment_lines.size(); i++) {
			const String &comment = comment_lines[i].replace("\n", "\n#. ");
			if (i == 0) {
				p_file->store_line("#. TRANSLATORS: " + comment);
			} else {
				p_file->store_line("#. " + comment);
			}
		}

		// Write file locations.
		const Vector<String> locations = p_template->get_hint(key.msgid, key.msgctxt, Translation::HINT_LOCATIONS).split("\n");
		for (const String &location : locations) {
			p_file->store_line("#: " + location.trim_prefix("res://").replace("\n", "\\n"));
		}

		// Write context.
		if (!key.msgctxt.is_empty()) {
			p_file->store_line("msgctxt " + String(key.msgctxt).json_escape().quote());
		}

		// Write msgid.
		_write_pot_field(p_file, "msgid", key.msgid);

		// Write msgid_plural.
		const String &msgid_plural = p_template->get_hint(key.msgid, key.msgctxt, Translation::HINT_PLURAL);
		if (msgid_plural.is_empty()) {
			p_file->store_line("msgstr \"\"");
		} else {
			_write_pot_field(p_file, "msgid_plural", msgid_plural);
			p_file->store_line("msgstr[0] \"\"");
			p_file->store_line("msgstr[1] \"\"");
		}
	}
}

void TranslationTemplateGenerator::_write_to_csv(Ref<FileAccess> p_file, const Ref<Translation> &p_template) const {
	List<Translation::MessageKey> keys;
	p_template->get_message_list(&keys);

	// Avoid adding unnecessary columns.
	bool context_used = false;
	bool plural_used = false;
	bool comments_used = false;
	bool locations_used = false;
	{
		for (const Translation::MessageKey &key : keys) {
			if (!context_used && !key.msgctxt.is_empty()) {
				context_used = true;
			}
			if (!plural_used && !p_template->get_hint(key.msgid, key.msgctxt, Translation::HINT_PLURAL).is_empty()) {
				plural_used = true;
			}
			if (!comments_used && !p_template->get_hint(key.msgid, key.msgctxt, Translation::HINT_COMMENTS).is_empty()) {
				comments_used = true;
			}
			if (!locations_used && !p_template->get_hint(key.msgid, key.msgctxt, Translation::HINT_LOCATIONS).is_empty()) {
				locations_used = true;
			}
		}
	}

	Vector<String> header = { "key" };
	if (context_used) {
		header.push_back("?context");
	}
	if (plural_used) {
		header.push_back("?plural");
	}
	if (comments_used) {
		header.push_back("_comments");
	}
	if (locations_used) {
		header.push_back("_locations");
	}
	p_file->store_csv_line(header);

	for (const Translation::MessageKey &key : keys) {
		Vector<String> line = { key.msgid };
		if (context_used) {
			line.push_back(key.msgctxt);
		}
		if (plural_used) {
			line.push_back(p_template->get_hint(key.msgid, key.msgctxt, Translation::HINT_PLURAL));
		}
		if (comments_used) {
			line.push_back(p_template->get_hint(key.msgid, key.msgctxt, Translation::HINT_COMMENTS));
		}
		if (locations_used) {
			line.push_back(p_template->get_hint(key.msgid, key.msgctxt, Translation::HINT_LOCATIONS));
		}
		p_file->store_csv_line(line);
	}
}

TranslationTemplateGenerator *TranslationTemplateGenerator::get_singleton() {
	if (!singleton) {
		singleton = memnew(TranslationTemplateGenerator);
	}
	return singleton;
}

TranslationTemplateGenerator::~TranslationTemplateGenerator() {
	memdelete(singleton);
	singleton = nullptr;
}
