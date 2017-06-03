/*************************************************************************/
/*  config_file.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "config_file.h"
#include "os/file_access.h"
#include "os/keyboard.h"
#include "variant_parser.h"

PoolStringArray ConfigFile::_get_sections() const {

	List<String> s;
	get_sections(&s);
	PoolStringArray arr;
	arr.resize(s.size());
	int idx = 0;
	for (const List<String>::Element *E = s.front(); E; E = E->next()) {

		arr.set(idx++, E->get());
	}

	return arr;
}

PoolStringArray ConfigFile::_get_section_keys(const String &p_section) const {

	List<String> s;
	get_section_keys(p_section, &s);
	PoolStringArray arr;
	arr.resize(s.size());
	int idx = 0;
	for (const List<String>::Element *E = s.front(); E; E = E->next()) {

		arr.set(idx++, E->get());
	}

	return arr;
}

void ConfigFile::set_value(const String &p_section, const String &p_key, const Variant &p_value) {

	if (p_value.get_type() == Variant::NIL) {
		//erase
		if (!values.has(p_section))
			return; // ?
		values[p_section].erase(p_key);
		if (values[p_section].empty()) {
			values.erase(p_section);
		}

	} else {
		if (!values.has(p_section)) {
			values[p_section] = Map<String, Variant>();
		}

		values[p_section][p_key] = p_value;
	}
}
Variant ConfigFile::get_value(const String &p_section, const String &p_key, Variant p_default) const {

	if (!values.has(p_section) || !values[p_section].has(p_key)) {
		if (p_default.get_type() == Variant::NIL) {
			ERR_EXPLAIN("Couldn't find the given section/key and no default was given");
			ERR_FAIL_V(p_default);
		}
		return p_default;
	}
	return values[p_section][p_key];
}

bool ConfigFile::has_section(const String &p_section) const {

	return values.has(p_section);
}
bool ConfigFile::has_section_key(const String &p_section, const String &p_key) const {

	if (!values.has(p_section))
		return false;
	return values[p_section].has(p_key);
}

void ConfigFile::get_sections(List<String> *r_sections) const {

	for (const Map<String, Map<String, Variant> >::Element *E = values.front(); E; E = E->next()) {
		r_sections->push_back(E->key());
	}
}
void ConfigFile::get_section_keys(const String &p_section, List<String> *r_keys) const {

	ERR_FAIL_COND(!values.has(p_section));

	for (const Map<String, Variant>::Element *E = values[p_section].front(); E; E = E->next()) {
		r_keys->push_back(E->key());
	}
}

void ConfigFile::erase_section(const String &p_section) {

	values.erase(p_section);
}

Error ConfigFile::save(const String &p_path) {

	Error err;
	FileAccess *file = FileAccess::open(p_path, FileAccess::WRITE, &err);

	if (err) {
		if (file)
			memdelete(file);
		return err;
	}

	for (Map<String, Map<String, Variant> >::Element *E = values.front(); E; E = E->next()) {

		if (E != values.front())
			file->store_string("\n");
		file->store_string("[" + E->key() + "]\n\n");

		for (Map<String, Variant>::Element *F = E->get().front(); F; F = F->next()) {

			String vstr;
			VariantWriter::write_to_string(F->get(), vstr);
			file->store_string(F->key() + "=" + vstr + "\n");
		}
	}

	memdelete(file);

	return OK;
}

Error ConfigFile::load(const String &p_path) {

	Error err;
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ, &err);

	if (!f)
		return ERR_CANT_OPEN;

	VariantParser::StreamFile stream;
	stream.f = f;

	String assign;
	Variant value;
	VariantParser::Tag next_tag;

	int lines = 0;
	String error_text;

	String section;

	while (true) {

		assign = Variant();
		next_tag.fields.clear();
		next_tag.name = String();

		err = VariantParser::parse_tag_assign_eof(&stream, lines, error_text, next_tag, assign, value, NULL, true);
		if (err == ERR_FILE_EOF) {
			memdelete(f);
			return OK;
		} else if (err != OK) {
			ERR_PRINTS("ConfgFile::load - " + p_path + ":" + itos(lines) + " error: " + error_text);
			memdelete(f);
			return err;
		}

		if (assign != String()) {
			set_value(section, assign, value);
		} else if (next_tag.name != String()) {
			section = next_tag.name;
		}
	}

	memdelete(f);

	return OK;
}

void ConfigFile::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_value", "section", "key", "value"), &ConfigFile::set_value);
	ClassDB::bind_method(D_METHOD("get_value:Variant", "section", "key", "default"), &ConfigFile::get_value, DEFVAL(Variant()));

	ClassDB::bind_method(D_METHOD("has_section", "section"), &ConfigFile::has_section);
	ClassDB::bind_method(D_METHOD("has_section_key", "section", "key"), &ConfigFile::has_section_key);

	ClassDB::bind_method(D_METHOD("get_sections"), &ConfigFile::_get_sections);
	ClassDB::bind_method(D_METHOD("get_section_keys", "section"), &ConfigFile::_get_section_keys);

	ClassDB::bind_method(D_METHOD("erase_section", "section"), &ConfigFile::erase_section);

	ClassDB::bind_method(D_METHOD("load:Error", "path"), &ConfigFile::load);
	ClassDB::bind_method(D_METHOD("save:Error", "path"), &ConfigFile::save);
}

ConfigFile::ConfigFile() {
}
