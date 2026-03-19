/**************************************************************************/
/*  resource_importer_csv_translation.cpp                                 */
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

#include "resource_importer_csv_translation.h"

#include "core/io/file_access.h"
#include "core/io/resource_saver.h"
#include "core/string/optimized_translation.h"
#include "core/string/translation_server.h"

String ResourceImporterCSVTranslation::get_importer_name() const {
	return "csv_translation";
}

String ResourceImporterCSVTranslation::get_visible_name() const {
	return "CSV Translation";
}

void ResourceImporterCSVTranslation::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("csv");
}

String ResourceImporterCSVTranslation::get_save_extension() const {
	return ""; //does not save a single resource
}

String ResourceImporterCSVTranslation::get_resource_type() const {
	return "Translation";
}

bool ResourceImporterCSVTranslation::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	return true;
}

int ResourceImporterCSVTranslation::get_preset_count() const {
	return 0;
}

String ResourceImporterCSVTranslation::get_preset_name(int p_idx) const {
	return "";
}

void ResourceImporterCSVTranslation::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "compress", PROPERTY_HINT_ENUM, "Disabled,Auto"), 1)); // Enum for compatibility with previous versions.
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "delimiter", PROPERTY_HINT_ENUM, "Comma,Semicolon,Tab"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "unescape_keys"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "unescape_translations"), true));
}

Error ResourceImporterCSVTranslation::import(ResourceUID::ID p_source_id, const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	Ref<FileAccess> f = FileAccess::open(p_source_file, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_INVALID_PARAMETER, "Cannot open file from path '" + p_source_file + "'.");

	String delimiter;
	switch ((int)p_options["delimiter"]) {
		case 1: {
			delimiter = ";";
		} break;
		case 2: {
			delimiter = "\t";
		} break;
		default: {
			delimiter = ",";
		} break;
	}

	// Parse the header row.
	HashMap<int, Ref<Translation>> column_to_translation;
	int context_column = -1;
	int plural_column = -1;
	{
		const Vector<String> line = f->get_csv_line(delimiter);
		for (int i = 1; i < line.size(); i++) {
			if (line[i].left(1) == "_") {
				continue;
			}
			if (line[i].to_lower() == "?context") {
				ERR_CONTINUE_MSG(context_column != -1, "Error importing CSV translation: Multiple '?context' columns found. Only one is allowed. Subsequent ones will be ignored.");
				context_column = i;
				continue;
			}
			if (line[i].to_lower() == "?plural") {
				ERR_CONTINUE_MSG(plural_column != -1, "Error importing CSV translation: Multiple '?plural' columns found. Only one is allowed. Subsequent ones will be ignored.");
				plural_column = i;
				continue;
			}

			const String locale = TranslationServer::get_singleton()->standardize_locale(line[i]);
			ERR_CONTINUE_MSG(locale.is_empty(), vformat("Error importing CSV translation: Invalid locale format '%s', should be 'language_Script_COUNTRY_VARIANT@extra'. This column will be ignored.", line[i]));

			Ref<Translation> translation;
			translation.instantiate();
			translation->set_locale(locale);
			column_to_translation[i] = translation;
		}

		if (column_to_translation.is_empty()) {
			WARN_PRINT(vformat("CSV file '%s' does not contain any translation.", p_source_file));
			return OK;
		}
	}

	// Parse content rows.
	bool context_used = false;
	bool plural_used = false;
	{
		const bool unescape_keys = p_options.has("unescape_keys") ? bool(p_options["unescape_keys"]) : false;
		const bool unescape_translations = p_options.has("unescape_translations") ? bool(p_options["unescape_translations"]) : true;

		bool reading_plural_rows = false;
		String plural_msgid;
		String plural_msgctxt;
		HashMap<int, Vector<String>> plural_msgstrs;

		do {
			const Vector<String> line = f->get_csv_line(delimiter);

			// Skip empty lines.
			if (line.size() == 1 && line[0].is_empty()) {
				continue;
			}

			if (line[0].to_lower() == "?pluralrule") {
				for (int i = 1; i < line.size(); i++) {
					if (line[i].is_empty() || !column_to_translation.has(i)) {
						continue;
					}
					Ref<Translation> translation = column_to_translation[i];
					ERR_CONTINUE_MSG(!translation->get_plural_rules_override().is_empty(), vformat("Error importing CSV translation: Multiple '?pluralrule' definitions found for locale '%s'. Only one is allowed. Subsequent ones will be ignored.", translation->get_locale()));
					translation->set_plural_rules_override(line[i]);
				}
				continue;
			}

			const String msgid = unescape_keys ? line[0].c_unescape() : line[0];
			if (!reading_plural_rows && msgid.is_empty()) {
				continue;
			}

			// It's okay if you define context or plural columns but don't use them.
			const String msgctxt = (context_column != -1 && context_column < line.size()) ? line[context_column] : String();
			if (!msgctxt.is_empty()) {
				context_used = true;
			}
			const String msgid_plural = (plural_column != -1 && plural_column < line.size()) ? line[plural_column] : String();
			if (!msgid_plural.is_empty()) {
				plural_used = true;
			}

			// End of plural rows.
			if (reading_plural_rows && (!msgid.is_empty() || !msgctxt.is_empty() || !msgid_plural.is_empty())) {
				reading_plural_rows = false;

				for (KeyValue<int, Ref<Translation>> E : column_to_translation) {
					Ref<Translation> translation = E.value;
					const Vector<String> &msgstrs = plural_msgstrs[E.key];
					if (!msgstrs.is_empty()) {
						translation->add_plural_message(plural_msgid, msgstrs, plural_msgctxt);
					}
				}
				plural_msgstrs.clear();
			}

			// Start of plural rows.
			if (!reading_plural_rows && !msgid_plural.is_empty()) {
				reading_plural_rows = true;
				plural_msgid = msgid;
				plural_msgctxt = msgctxt;
			}

			for (int i = 1; i < line.size(); i++) {
				if (!column_to_translation.has(i)) {
					continue;
				}
				const String msgstr = unescape_translations ? line[i].c_unescape() : line[i];
				if (msgstr.is_empty()) {
					continue;
				}
				if (reading_plural_rows) {
					plural_msgstrs[i].push_back(msgstr);
				} else {
					column_to_translation[i]->add_message(msgid, msgstr, msgctxt);
				}
			}
		} while (!f->eof_reached());

		if (reading_plural_rows) {
			for (KeyValue<int, Ref<Translation>> E : column_to_translation) {
				Ref<Translation> translation = E.value;
				const Vector<String> &msgstrs = plural_msgstrs[E.key];
				if (!msgstrs.is_empty()) {
					translation->add_plural_message(plural_msgid, msgstrs, plural_msgctxt);
				}
			}
		}
	}

	bool compress;
	switch ((int)p_options["compress"]) {
		case 0: { // Disabled.
			compress = false;
		} break;
		default: { // Auto.
			compress = !context_used && !plural_used;
		} break;
	}

	for (KeyValue<int, Ref<Translation>> E : column_to_translation) {
		Ref<Translation> xlt = E.value;

		if (compress) {
			Ref<OptimizedTranslation> cxl = memnew(OptimizedTranslation);
			cxl->generate(xlt);
			xlt = cxl;
		}

		String save_path = p_source_file.get_basename() + "." + xlt->get_locale() + ".translation";
		ResourceUID::ID save_id = hash64_murmur3_64(xlt->get_locale().hash64(), p_source_id) & 0x7FFFFFFFFFFFFFFF;
		bool uid_already_exists = ResourceUID::get_singleton()->has_id(save_id);
		if (uid_already_exists) {
			// Avoid creating a new file with a duplicate UID.
			// Always use this UID, even if the user has moved it to a different path.
			save_path = ResourceUID::get_singleton()->get_id_path(save_id);
		}

		ResourceSaver::save(xlt, save_path);
		if (r_gen_files) {
			r_gen_files->push_back(save_path);
		}
		if (!uid_already_exists) {
			// No need to call set_uid if save_path already refers to save_id.
			ResourceSaver::set_uid(save_path, save_id);
		}
	}

	return OK;
}
