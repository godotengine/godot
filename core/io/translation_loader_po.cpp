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

RES TranslationLoaderPO::load_translation(FileAccess *f, bool p_use_context, Error *r_error) {
	if (r_error) {
		*r_error = ERR_FILE_CORRUPT;
	}

	const String path = f->get_path();
	Ref<Translation> translation;
	if (p_use_context) {
		translation = Ref<Translation>(memnew(ContextTranslation));
	} else {
		translation.instance();
	}
	String config;

	uint32_t magic = f->get_32();
	if (magic == 0x950412de) {
		// Load binary MO file.

		uint16_t version_maj = f->get_16();
		uint16_t version_min = f->get_16();
		if (version_maj > 1) {
			ERR_FAIL_V_MSG(RES(), vformat("Unsupported MO file %s, version %d.%d.", path, version_maj, version_min));
		}

		uint32_t num_strings = f->get_32();
		uint32_t id_table_offset = f->get_32();
		uint32_t trans_table_offset = f->get_32();

		// Read string tables.
		for (uint32_t i = 0; i < num_strings; i++) {
			String msg_id;
			String msg_context;

			// Read id strings and context.
			{
				Vector<uint8_t> data;
				f->seek(id_table_offset + i * 8);
				uint32_t str_start = 0;
				uint32_t str_len = f->get_32();
				uint32_t str_offset = f->get_32();

				data.resize(str_len + 1);
				f->seek(str_offset);
				f->get_buffer(data.ptrw(), str_len);
				data.write[str_len] = 0;

				for (uint32_t j = 0; j < str_len + 1; j++) {
					if (data[j] == 0x04) {
						msg_context.parse_utf8((const char *)data.ptr(), j);
						str_start = j + 1;
					}
					if (data[j] == 0x00) {
						msg_id.parse_utf8((const char *)(data.ptr() + str_start), j - str_start);
						break;
					}
				}
			}

			// Read translated strings.
			{
				Vector<uint8_t> data;
				f->seek(trans_table_offset + i * 8);
				uint32_t str_len = f->get_32();
				uint32_t str_offset = f->get_32();

				data.resize(str_len + 1);
				f->seek(str_offset);
				f->get_buffer(data.ptrw(), str_len);
				data.write[str_len] = 0;

				if (msg_id.empty()) {
					config = String::utf8((const char *)data.ptr(), str_len);
				} else {
					for (uint32_t j = 0; j < str_len + 1; j++) {
						if (data[j] == 0x00) {
							translation->add_context_message(msg_id, String::utf8((const char *)data.ptr(), j), msg_context);
							break;
						}
					}
				}
			}
		}

		memdelete(f);
	} else {
		// Try to load as text PO file.
		f->seek(0);

		enum Status {
			STATUS_NONE,
			STATUS_READING_ID,
			STATUS_READING_STRING,
			STATUS_READING_CONTEXT,
		};

		Status status = STATUS_NONE;

		String msg_id;
		String msg_str;
		String msg_context;

		if (r_error) {
			*r_error = ERR_FILE_CORRUPT;
		}

		int line = 1;
		bool entered_context = false;
		bool skip_this = false;
		bool skip_next = false;
		bool is_eof = false;

		while (!is_eof) {
			String l = f->get_line().strip_edges();
			is_eof = f->eof_reached();

			// If we reached last line and it's not a content line, break, otherwise let processing that last loop
			if (is_eof && l.empty()) {
				if (status == STATUS_READING_ID || status == STATUS_READING_CONTEXT) {
					memdelete(f);
					ERR_FAIL_V_MSG(RES(), "Unexpected EOF while reading PO file at: " + path + ":" + itos(line));
				} else {
					break;
				}
			}

			if (l.begins_with("msgctxt")) {
				if (status != STATUS_READING_STRING) {
					memdelete(f);
					ERR_FAIL_V_MSG(RES(), "Unexpected 'msgctxt', was expecting 'msgstr' before 'msgctxt' while parsing: " + path + ":" + itos(line));
				}

				// In PO file, "msgctxt" appears before "msgid". If we encounter a "msgctxt", we add what we have read
				// and set "entered_context" to true to prevent adding twice.
				if (!skip_this && msg_id != "") {
					translation->add_context_message(msg_id, msg_str, msg_context);
				}
				msg_context = "";
				l = l.substr(7, l.length()).strip_edges();
				status = STATUS_READING_CONTEXT;
				entered_context = true;
			}

			if (l.begins_with("msgid")) {
				if (status == STATUS_READING_ID) {
					memdelete(f);
					ERR_FAIL_V_MSG(RES(), "Unexpected 'msgid', was expecting 'msgstr' while parsing: " + path + ":" + itos(line));
				}

				if (msg_id != "") {
					if (!skip_this && !entered_context) {
						translation->add_context_message(msg_id, msg_str, msg_context);
					}
				} else if (config == "") {
					config = msg_str;
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

			if (l.begins_with("msgstr")) {
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
			}

			line++;
		}

		memdelete(f);

		// Add the last set of data from last iteration.
		if (status == STATUS_READING_STRING) {
			if (msg_id != "") {
				if (!skip_this) {
					translation->add_context_message(msg_id, msg_str, msg_context);
				}
			} else if (config == "") {
				config = msg_str;
			}
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

	return load_translation(f, false, r_error);
}

void TranslationLoaderPO::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("po");
	p_extensions->push_back("mo");
}

bool TranslationLoaderPO::handles_type(const String &p_type) const {
	return (p_type == "Translation");
}

String TranslationLoaderPO::get_resource_type(const String &p_path) const {
	if (p_path.get_extension().to_lower() == "po" || p_path.get_extension().to_lower() == "mo") {
		return "Translation";
	}
	return "";
}

TranslationLoaderPO::TranslationLoaderPO() {
}
