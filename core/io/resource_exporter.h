/*************************************************************************/
/*  resource_exporter.h                                                  */
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

#ifndef RESOURCE_EXPORTER_H
#define RESOURCE_EXPORTER_H

#include "core/io/resource_loader.h"
#include "resource_saver.h"

class ResourceExporter;

class ResourceFormatExporter : public ResourceFormatSaver {

	struct PathAndType {
		String path;
		String type;
		String exporter;
		String group_file;
		Variant metadata;
	};

	Error _get_path_and_type(const String &p_path, PathAndType &r_path_and_type, bool *r_valid = NULL) const;

	static ResourceFormatExporter *singleton;

	//need them to stay in order to compute the settings hash
	struct SortExporterByName {
		bool operator()(const Ref<ResourceExporter> &p_a, const Ref<ResourceExporter> &p_b) const;
	};

	Vector<Ref<ResourceExporter> > exporters;

public:
	static ResourceFormatExporter *get_singleton() { return singleton; }
	virtual Error save(const String &p_path, const RES &p_resource, uint32_t p_flags = 0);
	virtual void get_recognized_exporter_extensions(List<String> *p_extensions) const;
	virtual void get_recognized_exporter_extensions_for_type(const String &p_type, List<String> *p_extensions) const;
	virtual bool recognize_path(const String &p_path) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
	virtual Variant get_resource_metadata(const String &p_path) const;
	virtual bool is_export_valid(const String &p_path) const;
	virtual void get_dependencies(const String &p_path, List<String> *p_dependencies, bool p_add_types = false);
	virtual bool is_exported(const String &p_path) const { return recognize_path(p_path); }
	//TODO
	virtual String get_export_group_file(const String &p_path) const;
	virtual bool exists(const String &p_path) const;
	//TODO
	virtual bool can_be_exported(const String &p_path) const;
	//TODO
	virtual int get_export_order(const String &p_path) const;

	String get_internal_resource_path(const String &p_path) const;
	void get_internal_resource_path_list(const String &p_path, List<String> *r_paths);

	void add_exporter(const Ref<ResourceExporter> &p_exporter) {
		exporters.push_back(p_exporter);
	}
	void remove_exporter(const Ref<ResourceExporter> &p_exporter) { exporters.erase(p_exporter); }
	Ref<ResourceExporter> get_exporter_by_name(const String &p_name) const;
	Ref<ResourceExporter> get_exporter_by_extension(const String &p_extension) const;
	void get_exporters_for_extension(const String &p_extension, List<Ref<ResourceExporter> > *r_exporters);

	bool are_export_settings_valid(const String &p_path) const;
	//TODO
	String get_export_settings_hash() const;
	//TODO
	String get_export_base_path(const String &p_for_file) const;
	ResourceFormatExporter();
	~ResourceFormatExporter() {}
};

class ResourceExporter : public Reference {

	GDCLASS(ResourceExporter, Reference);

public:
	virtual String get_exporter_name() const = 0;
	virtual String get_visible_name() const = 0;
	virtual void get_recognized_extensions(List<String> *p_extensions) const = 0;
	virtual String get_save_extension() const = 0;
	virtual String get_resource_type() const = 0;
	virtual float get_priority() const { return 1.0; }
	virtual int get_export_order() const { return 0; }

	struct ExportOption {
		PropertyInfo option;
		Variant default_value;

		ExportOption(const PropertyInfo &p_info, const Variant &p_default) :
				option(p_info),
				default_value(p_default) {
		}
		ExportOption() {}
	};

	virtual int get_preset_count() const { return 0; }
	virtual String get_preset_name(int p_idx) const { return String(); }

	virtual void get_export_options(List<ExportOption> *r_options, int p_preset = 0) const = 0;
	virtual bool get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const = 0;
	virtual String get_option_group_file() const { return String(); }

	virtual Error export_(Node *p_node, const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files = NULL, Variant *r_metadata = NULL) = 0;

	virtual Error export_group_file(const String &p_group_file, const Map<String, Map<StringName, Variant> > &p_source_file_options, const Map<String, String> &p_base_paths) { return ERR_UNAVAILABLE; }
	virtual bool are_export_settings_valid(const String &p_path) const { return true; }
	virtual String get_export_settings_string() const { return String(); }
};

#endif // RESOURCE_EXPORTER_H
