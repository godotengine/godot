/*************************************************************************/
/*  resource_import.h                                                    */
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
#ifndef RESOURCE_IMPORT_H
#define RESOURCE_IMPORT_H

#include "io/resource_loader.h"
class ResourceImporter;

class ResourceFormatImporter : public ResourceFormatLoader {

	struct PathAndType {
		String path;
		String type;
		String importer;
	};

	Error _get_path_and_type(const String &p_path, PathAndType &r_path_and_type, bool *r_valid = NULL) const;

	static ResourceFormatImporter *singleton;

	Set<Ref<ResourceImporter> > importers;

public:
	static ResourceFormatImporter *get_singleton() { return singleton; }
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = NULL);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual void get_recognized_extensions_for_type(const String &p_type, List<String> *p_extensions) const;
	virtual bool recognize_path(const String &p_path, const String &p_for_type = String()) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
	virtual bool is_import_valid(const String &p_path) const;
	virtual void get_dependencies(const String &p_path, List<String> *p_dependencies, bool p_add_types = false);

	virtual bool can_be_imported(const String &p_path) const;
	virtual int get_import_order(const String &p_path) const;

	String get_internal_resource_path(const String &p_path) const;
	void get_internal_resource_path_list(const String &p_path, List<String> *r_paths);

	void add_importer(const Ref<ResourceImporter> &p_importer) { importers.insert(p_importer); }
	void remove_importer(const Ref<ResourceImporter> &p_importer) { importers.erase(p_importer); }
	Ref<ResourceImporter> get_importer_by_name(const String &p_name) const;
	Ref<ResourceImporter> get_importer_by_extension(const String &p_extension) const;
	void get_importers_for_extension(const String &p_extension, List<Ref<ResourceImporter> > *r_importers);

	String get_import_base_path(const String &p_for_file) const;
	ResourceFormatImporter();
};

class ResourceImporter : public Reference {

	GDCLASS(ResourceImporter, Reference)
public:
	virtual String get_importer_name() const = 0;
	virtual String get_visible_name() const = 0;
	virtual void get_recognized_extensions(List<String> *p_extensions) const = 0;
	virtual String get_save_extension() const = 0;
	virtual String get_resource_type() const = 0;
	virtual float get_priority() const { return 1.0; }
	virtual int get_import_order() const { return 0; }

	struct ImportOption {
		PropertyInfo option;
		Variant default_value;

		ImportOption(const PropertyInfo &p_info, const Variant &p_default)
			: option(p_info),
			  default_value(p_default) {
		}
		ImportOption() {}
	};

	virtual int get_preset_count() const { return 0; }
	virtual String get_preset_name(int p_idx) const { return String(); }

	virtual void get_import_options(List<ImportOption> *r_options, int p_preset = 0) const = 0;
	virtual bool get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const = 0;

	virtual Error import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files = NULL) = 0;
};

#endif // RESOURCE_IMPORT_H
