/**************************************************************************/
/*  slime_ai_journal.cpp                                                */
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

#include "slime_ai_journal.h"

#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/json.h"

namespace SlimeAI {

Journal::Journal(const String &p_path) :
		path(p_path) {
	if (!FileAccess::exists(path)) {
		return;
	}
	Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
	if (file.is_null() || file->get_length() > 16 * 1024 * 1024) {
		load_error = "Journal unavailable or oversized.";
		return;
	}
	const String data = file->get_as_text();
	const PackedStringArray lines = data.split("\n", false);
	const bool terminated = data.ends_with("\n");
	for (int i = 0; i < lines.size(); i++) {
		// A crash can leave a torn final record. The preceding durable entry is
		// authoritative; future writes require operator review in that case.
		if (i == lines.size() - 1 && !terminated) {
			load_error = "Journal has a torn final record; inspect before continuing.";
			return;
		}
		Ref<JSON> json;
		json.instantiate();
		if (json->parse(lines[i]) != OK || json->get_data().get_type() != Variant::DICTIONARY) {
			load_error = "Journal contains an invalid record.";
			return;
		}
		const Dictionary line = json->get_data();
		if (line.size() != 3 || !line.has("operation_id") || !line.has("record_json") || !line.has("checksum") || line["operation_id"].get_type() != Variant::STRING || line["record_json"].get_type() != Variant::STRING || line["checksum"].get_type() != Variant::STRING) {
			load_error = "Journal record shape is invalid.";
			return;
		}
		const String id = line["operation_id"];
		const String record_json = line["record_json"];
		if (String(line["checksum"]) != (id + ":" + record_json).sha256_text()) {
			load_error = "Journal record checksum mismatch.";
			return;
		}
		Ref<JSON> record_parser;
		record_parser.instantiate();
		if (record_parser->parse(record_json) != OK || record_parser->get_data().get_type() != Variant::DICTIONARY) {
			load_error = "Journal payload is invalid.";
			return;
		}
		records[id] = record_parser->get_data();
	}
}

Dictionary Journal::get(const String &p_operation_id) const {
	return records.has(p_operation_id) && records[p_operation_id].get_type() == Variant::DICTIONARY ? Dictionary(records[p_operation_id]) : Dictionary();
}

Array Journal::operation_ids() const {
	return records.keys();
}

bool Journal::put(const String &p_operation_id, const Dictionary &p_record) {
	if (!valid() || p_operation_id.is_empty() || p_operation_id.length() > 128) {
		return false;
	}
	if (DirAccess::make_dir_recursive_absolute(path.get_base_dir()) != OK) {
		return false;
	}
	Dictionary line;
	line["operation_id"] = p_operation_id;
	const String record_json = JSON::stringify(p_record);
	line["record_json"] = record_json;
	line["checksum"] = (p_operation_id + ":" + record_json).sha256_text();
	Ref<FileAccess> file = FileAccess::open(path, FileAccess::exists(path) ? FileAccess::READ_WRITE : FileAccess::WRITE);
	if (file.is_null()) {
		return false;
	}
	file->seek_end();
	if (!file->store_string(JSON::stringify(line) + "\n")) {
		return false;
	}
	file->flush();
	records[p_operation_id] = p_record;
	return true;
}

} // namespace SlimeAI
