/*************************************************************************/
/*  resource_exporter.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "resource_exporter.h"

#include "core/os/os.h"
#include "core/variant_parser.h"

bool ResourceFormatExporter::SortExporterByName::operator()(const Ref<ResourceExporter> &p_a, const Ref<ResourceExporter> &p_b) const {
	return p_a->get_exporter_name() < p_b->get_exporter_name();
}

Error ResourceFormatExporter::_get_path_and_type(const String &p_path, PathAndType &r_path_and_type, bool *r_valid) const {

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
			ERR_PRINTS("ResourceFormatExporter::load - " + p_path + ".import:" + itos(lines) + " error: " + error_text);
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
			} else if (assign == "exporter") {
				r_path_and_type.exporter = value;
			} else if (assign == "group_file") {
				r_path_and_type.group_file = value;
			} else if (assign == "metadata") {
				r_path_and_type.metadata = value;
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
Error ResourceFormatExporter::save(const String &p_path, const RES &p_resource, uint32_t p_flags) {

	PathAndType pat;
	Error err = _get_path_and_type(p_path, pat);

	if (err != OK) {
		return err;
	}

	return ResourceSaver::save(pat.path, p_resource);
}

void ResourceFormatExporter::get_recognized_exporter_extensions(List<String> *p_extensions) const {
	Set<String> found;

	for (int i = 0; i < exporters.size(); i++) {
		List<String> local_exts;
		exporters[i]->get_recognized_extensions(&local_exts);
		for (List<String>::Element *F = local_exts.front(); F; F = F->next()) {
			if (!found.has(F->get())) {
				p_extensions->push_back(F->get());
				found.insert(F->get());
			}
		}
	}
}

void ResourceFormatExporter::get_recognized_exporter_extensions_for_type(const String &p_type, List<String> *p_extensions) const {

	if (p_type == "") {
		get_recognized_exporter_extensions(p_extensions);
		return;
	}

	Set<String> found;

	for (int i = 0; i < exporters.size(); i++) {
		String res_type = exporters[i]->get_resource_type();
		if (res_type == String())
			continue;

		if (!ClassDB::is_parent_class(res_type, p_type))
			continue;

		List<String> local_exts;
		exporters[i]->get_recognized_extensions(&local_exts);
		for (List<String>::Element *F = local_exts.front(); F; F = F->next()) {
			if (!found.has(F->get())) {
				p_extensions->push_back(F->get());
				found.insert(F->get());
			}
		}
	}
}

bool ResourceFormatExporter::exists(const String &p_path) const {

	return FileAccess::exists(p_path);
}

bool ResourceFormatExporter::recognize_path(const String &p_path) const {

	return FileAccess::exists(p_path);
}

bool ResourceFormatExporter::can_be_exported(const String &p_path) const {

	return ResourceFormatSaver::recognize_path(p_path);
}

int ResourceFormatExporter::get_export_order(const String &p_path) const {

	Ref<ResourceExporter> exporter;

	if (FileAccess::exists(p_path + ".import")) {

		PathAndType pat;
		Error err = _get_path_and_type(p_path, pat);

		if (err == OK) {
			exporter = get_exporter_by_name(pat.exporter);
		}
	} else {

		exporter = get_exporter_by_extension(p_path.get_extension().to_lower());
	}

	if (exporter.is_valid())
		return exporter->get_export_order();

	return 0;
}

bool ResourceFormatExporter::handles_type(const String &p_type) const {

	for (int i = 0; i < exporters.size(); i++) {

		String res_type = exporters[i]->get_resource_type();
		if (res_type == String())
			continue;
		if (ClassDB::is_parent_class(res_type, p_type))
			return true;
	}

	return true;
}

String ResourceFormatExporter::get_internal_resource_path(const String &p_path) const {

	PathAndType pat;
	Error err = _get_path_and_type(p_path, pat);

	if (err != OK) {

		return String();
	}

	return pat.path;
}

void ResourceFormatExporter::get_internal_resource_path_list(const String &p_path, List<String> *r_paths) {

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
			ERR_PRINTS("ResourceFormatExporter::get_internal_resource_path_list - " + p_path + ".import:" + itos(lines) + " error: " + error_text);
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

String ResourceFormatExporter::get_export_group_file(const String &p_path) const {

	bool valid = true;
	PathAndType pat;
	_get_path_and_type(p_path, pat, &valid);
	return valid ? pat.group_file : String();
}

bool ResourceFormatExporter::is_export_valid(const String &p_path) const {

	bool valid = true;
	PathAndType pat;
	_get_path_and_type(p_path, pat, &valid);
	return valid;
}

String ResourceFormatExporter::get_resource_type(const String &p_path) const {

	PathAndType pat;
	Error err = _get_path_and_type(p_path, pat);

	if (err != OK) {

		return "";
	}

	return pat.type;
}

Variant ResourceFormatExporter::get_resource_metadata(const String &p_path) const {
	PathAndType pat;
	Error err = _get_path_and_type(p_path, pat);

	if (err != OK) {

		return Variant();
	}

	return pat.metadata;
}

void ResourceFormatExporter::get_dependencies(const String &p_path, List<String> *p_dependencies, bool p_add_types) {

	PathAndType pat;
	Error err = _get_path_and_type(p_path, pat);

	if (err != OK) {

		return;
	}

	ResourceLoader::get_dependencies(pat.path, p_dependencies, p_add_types);
}

Ref<ResourceExporter> ResourceFormatExporter::get_exporter_by_name(const String &p_name) const {

	for (int i = 0; i < exporters.size(); i++) {
		if (exporters[i]->get_exporter_name() == p_name) {
			return exporters[i];
		}
	}

	return Ref<ResourceExporter>();
}

void ResourceFormatExporter::get_exporters_for_extension(const String &p_extension, List<Ref<ResourceExporter> > *r_exporters) {

	for (int i = 0; i < exporters.size(); i++) {
		List<String> local_exts;
		exporters[i]->get_recognized_extensions(&local_exts);
		for (List<String>::Element *F = local_exts.front(); F; F = F->next()) {
			if (p_extension.to_lower() == F->get()) {
				r_exporters->push_back(exporters[i]);
			}
		}
	}
}

Ref<ResourceExporter> ResourceFormatExporter::get_exporter_by_extension(const String &p_extension) const {

	Ref<ResourceExporter> exporter;
	float priority = 0;

	for (int i = 0; i < exporters.size(); i++) {

		List<String> local_exts;
		exporters[i]->get_recognized_extensions(&local_exts);
		for (List<String>::Element *F = local_exts.front(); F; F = F->next()) {
			if (p_extension.to_lower() == F->get() && exporters[i]->get_priority() > priority) {
				exporter = exporters[i];
				priority = exporters[i]->get_priority();
			}
		}
	}

	return exporter;
}

String ResourceFormatExporter::get_export_base_path(const String &p_for_file) const {

	return "res://.import/" + p_for_file.get_file() + "-" + p_for_file.md5_text();
}

bool ResourceFormatExporter::are_export_settings_valid(const String &p_path) const {

	bool valid = true;
	PathAndType pat;
	_get_path_and_type(p_path, pat, &valid);

	if (!valid) {
		return false;
	}

	for (int i = 0; i < exporters.size(); i++) {
		if (exporters[i]->get_exporter_name() == pat.exporter) {
			if (!exporters[i]->are_export_settings_valid(p_path)) { //importer thinks this is not valid
				return false;
			}
		}
	}

	return true;
}

String ResourceFormatExporter::get_export_settings_hash() const {

	Vector<Ref<ResourceExporter> > sorted_exporters = exporters;

	sorted_exporters.sort_custom<SortExporterByName>();

	String hash;
	for (int i = 0; i < sorted_exporters.size(); i++) {
		hash += ":" + sorted_exporters[i]->get_exporter_name() + ":" + sorted_exporters[i]->get_export_settings_string();
	}
	return hash.md5_text();
}

ResourceFormatExporter *ResourceFormatExporter::singleton = NULL;

ResourceFormatExporter::ResourceFormatExporter() {
	singleton = this;
}
