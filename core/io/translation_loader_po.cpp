/*************************************************************************/
/*  translation_loader_po.cpp                                            */
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

#include "translation_loader_po.h"

#include "core/os/file_access.h"
#include "core/translation.h"
#include "core/translation_po.h"

RES TranslationLoaderPO::load_translation(FileAccess *f, Error *r_error) {
	enum Status {
		STATUS_NONE,
		STATUS_READING_ID,
		STATUS_READING_STRING,
		STATUS_READING_CONTEXT,
		STATUS_READING_PLURAL,
	};

	Status status = STATUS_NONE;

	String msg_id;
	String msg_str;
	String msg_context;
	Vector<String> msgs_plural;
	String config;

	if (r_error) {
		*r_error = ERR_FILE_CORRUPT;
	}

	Ref<TranslationPO> translation = Ref<TranslationPO>(memnew(TranslationPO));
	int line = 1;
	int plural_forms = 0;
	int plural_index = -1;
	bool entered_context = false;
	bool skip_this = false;
	bool skip_next = false;
	bool is_eof = false;
	const String path = f->get_path();

	while (!is_eof) {
		String l = f->get_line().strip_edges();
		is_eof = f->eof_reached();

		// If we reached last line and it's not a content line, break, otherwise let processing that last loop
		if (is_eof && l.empty()) {
			if (status == STATUS_READING_ID || status == STATUS_READING_CONTEXT || (status == STATUS_READING_PLURAL && plural_index != plural_forms - 1)) {
				memdelete(f);
				ERR_FAIL_V_MSG(RES(), "Unexpected EOF while reading PO file at: " + path + ":" + itos(line));
			} else {
				break;
			}
		}

		if (l.begins_with("msgctxt")) {
			if (status != STATUS_READING_STRING && status != STATUS_READING_PLURAL) {
				memdelete(f);
				ERR_FAIL_V_MSG(RES(), "Unexpected 'msgctxt', was expecting 'msgid_plural' or 'msgstr' before 'msgctxt' while parsing: " + path + ":" + itos(line));
			}

			// In PO file, "msgctxt" appears before "msgid". If we encounter a "msgctxt", we add what we have read
			// and set "entered_context" to true to prevent adding twice.
			if (!skip_this && msg_id != "") {
				if (status == STATUS_READING_STRING) {
					translation->add_message(msg_id, msg_str, msg_context);
				} else if (status == STATUS_READING_PLURAL) {
					if (plural_index != plural_forms - 1) {
						memdelete(f);
						ERR_FAIL_V_MSG(RES(), "Number of 'msgstr[]' doesn't match with number of plural forms: " + path + ":" + itos(line));
					}
					translation->add_plural_message(msg_id, msgs_plural, msg_context);
				}
			}
			msg_context = "";
			l = l.substr(7, l.length()).strip_edges();
			status = STATUS_READING_CONTEXT;
			entered_context = true;
		}

		if (l.begins_with("msgid_plural")) {
			if (plural_forms == 0) {
				memdelete(f);
				ERR_FAIL_V_MSG(RES(), "PO file uses 'msgid_plural' but 'Plural-Forms' is invalid or missing in header: " + path + ":" + itos(line));
			} else if (status != STATUS_READING_ID) {
				memdelete(f);
				ERR_FAIL_V_MSG(RES(), "Unexpected 'msgid_plural', was expecting 'msgid' before 'msgid_plural' while parsing: " + path + ":" + itos(line));
			}
			// We don't record the message in "msgid_plural" itself as tr_n(), TTRN(), RTRN() interfaces provide the plural string already.
			// We just have to reset variables related to plurals for "msgstr[]" later on.
			l = l.substr(12, l.length()).strip_edges();
			plural_index = -1;
			msgs_plural.clear();
			msgs_plural.resize(plural_forms);
			status = STATUS_READING_PLURAL;
		} else if (l.begins_with("msgid")) {
			if (status == STATUS_READING_ID) {
				memdelete(f);
				ERR_FAIL_V_MSG(RES(), "Unexpected 'msgid', was expecting 'msgstr' while parsing: " + path + ":" + itos(line));
			}

			if (msg_id != "") {
				if (!skip_this && !entered_context) {
					if (status == STATUS_READING_STRING) {
						translation->add_message(msg_id, msg_str, msg_context);
					} else if (status == STATUS_READING_PLURAL) {
						if (plural_index != plural_forms - 1) {
							memdelete(f);
							ERR_FAIL_V_MSG(RES(), "Number of 'msgstr[]' doesn't match with number of plural forms: " + path + ":" + itos(line));
						}
						translation->add_plural_message(msg_id, msgs_plural, msg_context);
					}
				}
			} else if (config == "") {
				config = msg_str;
				// Record plural rule.
				int p_start = config.find("Plural-Forms");
				if (p_start != -1) {
					int p_end = config.find("\n", p_start);
					translation->set_plural_rule(config.substr(p_start, p_end - p_start));
					plural_forms = translation->get_plural_forms();
				}
			}

			l = l.substr(5, l.length()).strip_edges();
			status = STATUS_READING_ID;
			// If we did not encounter msgctxt, we reset context to empty to reset it.
			if (!entered_context) {
				msg_context = "";
			}
			msg_id = "";
			msg_str = "";
			skip_this = skip_next;
			skip_next = false;
			entered_context = false;
		}

		if (l.begins_with("msgstr[")) {
			if (status != STATUS_READING_PLURAL) {
				memdelete(f);
				ERR_FAIL_V_MSG(RES(), "Unexpected 'msgstr[]', was expecting 'msgid_plural' before 'msgstr[]' while parsing: " + path + ":" + itos(line));
			}
			plural_index++; // Increment to add to the next slot in vector msgs_plural.
			l = l.substr(9, l.length()).strip_edges();
		} else if (l.begins_with("msgstr")) {
			if (status != STATUS_READING_ID) {
				memdelete(f);
				ERR_FAIL_V_MSG(RES(), "Unexpected 'msgstr', was expecting 'msgid' before 'msgstr' while parsing: " + path + ":" + itos(line));
			}

			l = l.substr(6, l.length()).strip_edges();
			status = STATUS_READING_STRING;
		}

		if (l == "" || l.begins_with("#")) {
			if (l.find("fuzzy") != -1) {
				skip_next = true;
			}
			line++;
			continue; // Nothing to read or comment.
		}

		if (!l.begins_with("\"") || status == STATUS_NONE) {
			memdelete(f);
			ERR_FAIL_V_MSG(RES(), "Invalid line '" + l + "' while parsing: " + path + ":" + itos(line));
		}

		l = l.substr(1, l.length());
		// Find final quote, ignoring escaped ones (\").
		// The escape_next logic is necessary to properly parse things like \\"
		// where the blackslash is the one being escaped, not the quote.
		int end_pos = -1;
		bool escape_next = false;
		for (int i = 0; i < l.length(); i++) {
			if (l[i] == '\\' && !escape_next) {
				escape_next = true;
				continue;
			}

			if (l[i] == '"' && !escape_next) {
				end_pos = i;
				break;
			}

			escape_next = false;
		}

		if (end_pos == -1) {
			memdelete(f);
			ERR_FAIL_V_MSG(RES(), "Expected '\"' at end of message while parsing: " + path + ":" + itos(line));
		}

		l = l.substr(0, end_pos);
		l = l.c_unescape();

		if (status == STATUS_READING_ID) {
			msg_id += l;
		} else if (status == STATUS_READING_STRING) {
			msg_str += l;
		} else if (status == STATUS_READING_CONTEXT) {
			msg_context += l;
		} else if (status == STATUS_READING_PLURAL && plural_index >= 0) {
			msgs_plural.write[plural_index] = msgs_plural[plural_index] + l;
		}

		line++;
	}

	memdelete(f);

	// Add the last set of data from last iteration.
	if (status == STATUS_READING_STRING) {
		if (msg_id != "") {
			if (!skip_this) {
				translation->add_message(msg_id, msg_str, msg_context);
			}
		} else if (config == "") {
			config = msg_str;
		}
	} else if (status == STATUS_READING_PLURAL) {
		if (!skip_this && msg_id != "") {
			if (plural_index != plural_forms - 1) {
				memdelete(f);
				ERR_FAIL_V_MSG(RES(), "Number of 'msgstr[]' doesn't match with number of plural forms: " + path + ":" + itos(line));
			}
			translation->add_plural_message(msg_id, msgs_plural, msg_context);
		}
	}

	ERR_FAIL_COND_V_MSG(config == "", RES(), "No config found in file: " + path + ".");

	Vector<String> configs = config.split("\n");
	for (int i = 0; i < configs.size(); i++) {
		String c = configs[i].strip_edges();
		int p = c.find(":");
		if (p == -1) {
			continue;
		}
		String prop = c.substr(0, p).strip_edges();
		String value = c.substr(p + 1, c.length()).strip_edges();

		if (prop == "X-Language" || prop == "Language") {
			translation->set_locale(value);
		}
	}

	if (r_error) {
		*r_error = OK;
	}

	return translation;
}

RES TranslationLoaderPO::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, bool p_no_cache) {
	if (r_error) {
		*r_error = ERR_CANT_OPEN;
	}

	FileAccess *f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(!f, RES(), "Cannot open file '" + p_path + "'.");

	return load_translation(f, r_error);
}

void TranslationLoaderPO::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("po");
}

bool TranslationLoaderPO::handles_type(const String &p_type) const {
	return (p_type == "Translation");
}

String TranslationLoaderPO::get_resource_type(const String &p_path) const {
	if (p_path.get_extension().to_lower() == "po") {
		return "Translation";
	}
	return "";
}
