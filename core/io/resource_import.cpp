/*************************************************************************/
/*  resource_import.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "resource_import.h"

#include "os/os.h"
#include "variant_parser.h"

Error ResourceFormatImporter::_get_path_and_type(const String &p_path, PathAndType &r_path_and_type, bool *r_valid) const {

	Error err;
	FileAccess *f = FileAccess::open(p_path + ".import", FileAccess::READ, &err);

	if (!f) {
		if (r_valid) {
			*r_valid = false;
		}
		return err;
	}

	VariantParser::StreamFile stream;
	stream.f = f;

	String assign;
	Variant value;
	VariantParser::Tag next_tag;

	if (r_valid) {
		*r_valid = true;
	}

	int lines = 0;
	String error_text;
	bool path_found = false; //first match must have priority
	while (true) {

		assign = Variant();
		next_tag.fields.clear();
		next_tag.name = String();

		err = VariantParser::parse_tag_assign_eof(&stream, lines, error_text, next_tag, assign, value, NULL, true);
		if (err == ERR_FILE_EOF) {
			memdelete(f);
			return OK;
		} else if (err != OK) {
			ERR_PRINTS("ResourceFormatImporter::load - " + p_path + ".import:" + itos(lines) + " error: " + error_text);
			memdelete(f);
			return err;
		}

		if (assign != String()) {
			if (!path_found && assign.begins_with("path.") && r_path_and_type.path == String()) {
				String feature = assign.get_slicec('.', 1);
				if (OS::get_singleton()->has_feature(feature)) {
					r_path_and_type.path = value;
					path_found = true; //first match must have priority
				}

			} else if (!path_found && assign == "path") {
				r_path_and_type.path = value;
				path_found = true; //first match must have priority
			} else if (assign == "type") {
				r_path_and_type.type = value;
			} else if (assign == "importer") {
				r_path_and_type.importer = value;
			} else if (assign == "valid") {
				if (r_valid) {
					*r_valid = value;
				}
			}

		} else if (next_tag.name != "remap") {
			break;
		}
	}

	memdelete(f);

	if (r_path_and_type.path == String() || r_path_and_type.type == String()) {
		return ERR_FILE_CORRUPT;
	}
	return OK;
}

RES ResourceFormatImporter::load(const String &p_path, const String &p_original_path, Error *r_error) {

	PathAndType pat;
	Error err = _get_path_and_type(p_path, pat);

	if (err != OK) {

		if (r_error)
			*r_error = err;

		return RES();
	}

	RES res = ResourceLoader::_load(pat.path, p_path, pat.type, false, r_error);

#ifdef TOOLS_ENABLED
	if (res.is_valid()) {
		res->set_import_last_modified_time(res->get_last_modified_time()); //pass this, if used
		res->set_import_path(pat.path);
	}
#endif

	return res;
}

void ResourceFormatImporter::get_recognized_extensions(List<String> *p_extensions) const {

	Set<String> found;

	for (Set<Ref<ResourceImporter> >::Element *E = importers.front(); E; E = E->next()) {
		List<String> local_exts;
		E->get()->get_recognized_extensions(&local_exts);
		for (List<String>::Element *F = local_exts.front(); F; F = F->next()) {
			if (!found.has(F->get())) {
				p_extensions->push_back(F->get());
				found.insert(F->get());
			}
		}
	}
}

void ResourceFormatImporter::get_recognized_extensions_for_type(const String &p_type, List<String> *p_extensions) const {

	if (p_type == "") {
		return get_recognized_extensions(p_extensions);
	}

	Set<String> found;

	for (Set<Ref<ResourceImporter> >::Element *E = importers.front(); E; E = E->next()) {
		String res_type = E->get()->get_resource_type();
		if (res_type == String())
			continue;

		if (!ClassDB::is_parent_class(res_type, p_type))
			continue;

		List<String> local_exts;
		E->get()->get_recognized_extensions(&local_exts);
		for (List<String>::Element *F = local_exts.front(); F; F = F->next()) {
			if (!found.has(F->get())) {
				p_extensions->push_back(F->get());
				found.insert(F->get());
			}
		}
	}
}

bool ResourceFormatImporter::recognize_path(const String &p_path, const String &p_for_type) const {

	return FileAccess::exists(p_path + ".import");
}

bool ResourceFormatImporter::can_be_imported(const String &p_path) const {

	return ResourceFormatLoader::recognize_path(p_path);
}

int ResourceFormatImporter::get_import_order(const String &p_path) const {

	Ref<ResourceImporter> importer;

	if (FileAccess::exists(p_path + ".import")) {

		PathAndType pat;
		Error err = _get_path_and_type(p_path, pat);

		if (err == OK) {
			importer = get_importer_by_name(pat.importer);
		}
	} else {

		importer = get_importer_by_extension(p_path.get_extension().to_lower());
	}

	if (importer.is_valid())
		return importer->get_import_order();

	return 0;
}

bool ResourceFormatImporter::handles_type(const String &p_type) const {

	for (Set<Ref<ResourceImporter> >::Element *E = importers.front(); E; E = E->next()) {

		String res_type = E->get()->get_resource_type();
		if (res_type == String())
			continue;
		if (ClassDB::is_parent_class(res_type, p_type))
			return true;
	}

	return true;
}

String ResourceFormatImporter::get_internal_resource_path(const String &p_path) const {

	PathAndType pat;
	Error err = _get_path_and_type(p_path, pat);

	if (err != OK) {

		return String();
	}

	return pat.path;
}

void ResourceFormatImporter::get_internal_resource_path_list(const String &p_path, List<String> *r_paths) {

	Error err;
	FileAccess *f = FileAccess::open(p_path + ".import", FileAccess::READ, &err);

	if (!f)
		return;

	VariantParser::StreamFile stream;
	stream.f = f;

	String assign;
	Variant value;
	VariantParser::Tag next_tag;

	int lines = 0;
	String error_text;
	while (true) {

		assign = Variant();
		next_tag.fields.clear();
		next_tag.name = String();

		err = VariantParser::parse_tag_assign_eof(&stream, lines, error_text, next_tag, assign, value, NULL, true);
		if (err == ERR_FILE_EOF) {
			memdelete(f);
			return;
		} else if (err != OK) {
			ERR_PRINTS("ResourceFormatImporter::get_internal_resource_path_list - " + p_path + ".import:" + itos(lines) + " error: " + error_text);
			memdelete(f);
			return;
		}

		if (assign != String()) {
			if (assign.begins_with("path.")) {
				r_paths->push_back(value);
			} else if (assign == "path") {
				r_paths->push_back(value);
			}
		} else if (next_tag.name != "remap") {
			break;
		}
	}
	memdelete(f);
}

bool ResourceFormatImporter::is_import_valid(const String &p_path) const {

	bool valid = true;
	PathAndType pat;
	_get_path_and_type(p_path, pat, &valid);
	return valid;
}

String ResourceFormatImporter::get_resource_type(const String &p_path) const {

	PathAndType pat;
	Error err = _get_path_and_type(p_path, pat);

	if (err != OK) {

		return "";
	}

	return pat.type;
}

void ResourceFormatImporter::get_dependencies(const String &p_path, List<String> *p_dependencies, bool p_add_types) {

	PathAndType pat;
	Error err = _get_path_and_type(p_path, pat);

	if (err != OK) {

		return;
	}

	return ResourceLoader::get_dependencies(pat.path, p_dependencies, p_add_types);
}

Ref<ResourceImporter> ResourceFormatImporter::get_importer_by_name(const String &p_name) const {

	for (Set<Ref<ResourceImporter> >::Element *E = importers.front(); E; E = E->next()) {
		if (E->get()->get_importer_name() == p_name) {
			return E->get();
		}
	}

	return Ref<ResourceImporter>();
}

void ResourceFormatImporter::get_importers_for_extension(const String &p_extension, List<Ref<ResourceImporter> > *r_importers) {

	for (Set<Ref<ResourceImporter> >::Element *E = importers.front(); E; E = E->next()) {
		List<String> local_exts;
		E->get()->get_recognized_extensions(&local_exts);
		for (List<String>::Element *F = local_exts.front(); F; F = F->next()) {
			if (p_extension.to_lower() == F->get()) {
				r_importers->push_back(E->get());
			}
		}
	}
}

Ref<ResourceImporter> ResourceFormatImporter::get_importer_by_extension(const String &p_extension) const {

	Ref<ResourceImporter> importer;
	float priority = 0;

	for (Set<Ref<ResourceImporter> >::Element *E = importers.front(); E; E = E->next()) {

		List<String> local_exts;
		E->get()->get_recognized_extensions(&local_exts);
		for (List<String>::Element *F = local_exts.front(); F; F = F->next()) {
			if (p_extension.to_lower() == F->get() && E->get()->get_priority() > priority) {
				importer = E->get();
				priority = E->get()->get_priority();
			}
		}
	}

	return importer;
}

String ResourceFormatImporter::get_import_base_path(const String &p_for_file) const {

	return "res://.import/" + p_for_file.get_file() + "-" + p_for_file.md5_text();
}

ResourceFormatImporter *ResourceFormatImporter::singleton = NULL;

ResourceFormatImporter::ResourceFormatImporter() {
	singleton = this;
}
