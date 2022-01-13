/*************************************************************************/
/*  translation_loader_po.cpp                                            */
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

#include "translation_loader_po.h"

#include "core/os/file_access.h"
#include "core/translation.h"

RES TranslationLoaderPO::load_translation(FileAccess *f, Error *r_error) {
	enum Status {
		STATUS_NONE,
		STATUS_READING_ID,
		STATUS_READING_STRING,
	};

	Status status = STATUS_NONE;

	String msg_id;
	String msg_str;
	String config;

	if (r_error) {
		*r_error = ERR_FILE_CORRUPT;
	}

	Ref<Translation> translation = Ref<Translation>(memnew(Translation));
	int line = 1;
	bool skip_this = false;
	bool skip_next = false;
	bool is_eof = false;
	const String path = f->get_path();

	while (!is_eof) {
		String l = f->get_line().strip_edges();
		is_eof = f->eof_reached();

		// If we reached last line and it's not a content line, break, otherwise let processing that last loop
		if (is_eof && l.empty()) {
			if (status == STATUS_READING_ID) {
				memdelete(f);
				ERR_FAIL_V_MSG(RES(), "Unexpected EOF while reading 'msgid' at: " + path + ":" + itos(line));
			} else {
				break;
			}
		}

		if (l.begins_with("msgid")) {
			if (status == STATUS_READING_ID) {
				memdelete(f);
				ERR_FAIL_V_MSG(RES(), "Unexpected 'msgid', was expecting 'msgstr' while parsing: " + path + ":" + itos(line));
			}

			if (msg_id != "") {
				if (!skip_this) {
					translation->add_message(msg_id, msg_str);
				}
			} else if (config == "") {
				config = msg_str;
			}

			l = l.substr(5, l.length()).strip_edges();
			status = STATUS_READING_ID;
			msg_id = "";
			msg_str = "";
			skip_this = skip_next;
			skip_next = false;
		}

		if (l.begins_with("msgstr")) {
			if (status != STATUS_READING_ID) {
				memdelete(f);
				ERR_FAIL_V_MSG(RES(), "Unexpected 'msgstr', was expecting 'msgid' while parsing: " + path + ":" + itos(line));
			}

			l = l.substr(6, l.length()).strip_edges();
			status = STATUS_READING_STRING;
		}

		if (l == "" || l.begins_with("#")) {
			if (l.find("fuzzy") != -1) {
				skip_next = true;
			}
			line++;
			continue; //nothing to read or comment
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
		} else {
			msg_str += l;
		}

		line++;
	}

	memdelete(f);

	if (status == STATUS_READING_STRING) {
		if (msg_id != "") {
			if (!skip_this) {
				translation->add_message(msg_id, msg_str);
			}
		} else if (config == "") {
			config = msg_str;
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

RES TranslationLoaderPO::load(const String &p_path, const String &p_original_path, Error *r_error) {
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

TranslationLoaderPO::TranslationLoaderPO() {
}
