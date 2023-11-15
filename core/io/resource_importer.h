/**************************************************************************/
/*  resource_importer.h                                                   */
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

#ifndef RESOURCE_IMPORTER_H
#define RESOURCE_IMPORTER_H

#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"

class ResourceImporter;

class ResourceFormatImporter : public ResourceFormatLoader {
	struct PathAndType {
		String path;
		String type;
		String importer;
		String group_file;
		Variant metadata;
		uint64_t uid = ResourceUID::INVALID_ID;
	};

	Error _get_path_and_type(const String &p_path, PathAndType &r_path_and_type, bool *r_valid = nullptr) const;

	static ResourceFormatImporter *singleton;

	//need them to stay in order to compute the settings hash
	struct SortImporterByName {
		bool operator()(const Ref<ResourceImporter> &p_a, const Ref<ResourceImporter> &p_b) const;
	};

	Vector<Ref<ResourceImporter>> importers;

public:
	static ResourceFormatImporter *get_singleton() { return singleton; }
	virtual Ref<Resource> load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual void get_recognized_extensions_for_type(const String &p_type, List<String> *p_extensions) const;
	virtual bool recognize_path(const String &p_path, const String &p_for_type = String()) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
	virtual ResourceUID::ID get_resource_uid(const String &p_path) const;
	virtual Variant get_resource_metadata(const String &p_path) const;
	virtual bool is_import_valid(const String &p_path) const;
	virtual void get_dependencies(const String &p_path, List<String> *p_dependencies, bool p_add_types = false);
	virtual bool is_imported(const String &p_path) const { return recognize_path(p_path); }
	virtual String get_import_group_file(const String &p_path) const;
	virtual void get_classes_used(const String &p_path, HashSet<StringName> *r_classes);
	virtual bool exists(const String &p_path) const;

	virtual int get_import_order(const String &p_path) const;

	Error get_import_order_threads_and_importer(const String &p_path, int &r_order, bool &r_can_threads, String &r_importer) const;

	String get_internal_resource_path(const String &p_path) const;
	void get_internal_resource_path_list(const String &p_path, List<String> *r_paths);

	void add_importer(const Ref<ResourceImporter> &p_importer, bool p_first_priority = false);

	void remove_importer(const Ref<ResourceImporter> &p_importer) { importers.erase(p_importer); }
	Ref<ResourceImporter> get_importer_by_name(const String &p_name) const;
	Ref<ResourceImporter> get_importer_by_extension(const String &p_extension) const;
	void get_importers_for_extension(const String &p_extension, List<Ref<ResourceImporter>> *r_importers);
	void get_importers(List<Ref<ResourceImporter>> *r_importers);

	bool are_import_settings_valid(const String &p_path) const;
	String get_import_settings_hash() const;

	String get_import_base_path(const String &p_for_file) const;
	ResourceFormatImporter();
};

class ResourceImporter : public RefCounted {
	GDCLASS(ResourceImporter, RefCounted);

protected:
	static void _bind_methods();

public:
	virtual String get_importer_name() const = 0;
	virtual String get_visible_name() const = 0;
	virtual void get_recognized_extensions(List<String> *p_extensions) const = 0;
	virtual String get_save_extension() const = 0;
	virtual String get_resource_type() const = 0;
	virtual float get_priority() const { return 1.0; }
	virtual int get_import_order() const { return IMPORT_ORDER_DEFAULT; }
	virtual int get_format_version() const { return 0; }

	struct ImportOption {
		PropertyInfo option;
		Variant default_value;

		ImportOption(const PropertyInfo &p_info, const Variant &p_default) :
				option(p_info),
				default_value(p_default) {
		}
		ImportOption() {}
	};

	enum ImportOrder {
		IMPORT_ORDER_DEFAULT = 0,
		IMPORT_ORDER_SCENE = 100,
	};

	virtual bool has_advanced_options() const { return false; }
	virtual void show_advanced_options(const String &p_path) {}

	virtual int get_preset_count() const { return 0; }
	virtual String get_preset_name(int p_idx) const { return String(); }

	virtual void get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset = 0) const = 0;
	virtual bool get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const = 0;
	virtual void handle_compatibility_options(HashMap<StringName, Variant> &p_import_params) const {}
	virtual String get_option_group_file() const { return String(); }

	virtual Error import(const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files = nullptr, Variant *r_metadata = nullptr) = 0;
	virtual bool can_import_threaded() const { return true; }
	virtual void import_threaded_begin() {}
	virtual void import_threaded_end() {}

	virtual Error import_group_file(const String &p_group_file, const HashMap<String, HashMap<StringName, Variant>> &p_source_file_options, const HashMap<String, String> &p_base_paths) { return ERR_UNAVAILABLE; }
	virtual bool are_import_settings_valid(const String &p_path) const { return true; }
	virtual String get_import_settings_string() const { return String(); }
};

VARIANT_ENUM_CAST(ResourceImporter::ImportOrder);

class ResourceFormatImporterSaver : public ResourceFormatSaver {
	GDCLASS(ResourceFormatImporterSaver, ResourceFormatSaver)

public:
	virtual Error set_uid(const String &p_path, ResourceUID::ID p_uid) override;
};

#endif // RESOURCE_IMPORTER_H
