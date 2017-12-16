/*************************************************************************/
/*  editor_export.h                                                      */
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
#ifndef EDITOR_EXPORT_H
#define EDITOR_EXPORT_H

#include "os/dir_access.h"
#include "resource.h"
#include "scene/main/node.h"
#include "scene/main/timer.h"
#include "scene/resources/texture.h"

class EditorProgress;
class FileAccess;
class EditorExportPlatform;
class EditorFileSystemDirectory;

class EditorExportPreset : public Reference {

	GDCLASS(EditorExportPreset, Reference)
public:
	enum ExportFilter {
		EXPORT_ALL_RESOURCES,
		EXPORT_SELECTED_SCENES,
		EXPORT_SELECTED_RESOURCES,
	};

private:
	Ref<EditorExportPlatform> platform;
	ExportFilter export_filter;
	String include_filter;
	String exclude_filter;

	String exporter;
	Set<String> selected_files;
	bool runnable;

	Vector<String> patches;

	friend class EditorExport;
	friend class EditorExportPlatform;

	List<PropertyInfo> properties;
	Map<StringName, Variant> values;

	String name;

	String custom_features;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	Ref<EditorExportPlatform> get_platform() const;

	bool has(const StringName &p_property) const { return values.has(p_property); }

	Vector<String> get_files_to_export() const;

	void add_export_file(const String &p_path);
	void remove_export_file(const String &p_path);
	bool has_export_file(const String &p_path);

	void set_name(const String &p_name);
	String get_name() const;

	void set_runnable(bool p_enable);
	bool is_runnable() const;

	void set_export_filter(ExportFilter p_filter);
	ExportFilter get_export_filter() const;

	void set_include_filter(const String &p_include);
	String get_include_filter() const;

	void set_exclude_filter(const String &p_exclude);
	String get_exclude_filter() const;

	void add_patch(const String &p_path, int p_at_pos = -1);
	void set_patch(int p_index, const String &p_path);
	String get_patch(int p_index);
	void remove_patch(int p_idx);
	Vector<String> get_patches() const;

	void set_custom_features(const String &p_custom_features);
	String get_custom_features() const;

	const List<PropertyInfo> &get_properties() const { return properties; }

	EditorExportPreset();
};

struct SharedObject {
	String path;
	Vector<String> tags;

	SharedObject(const String &p_path, const Vector<String> &p_tags) :
			path(p_path),
			tags(p_tags) {
	}

	SharedObject() {}
};

class EditorExportPlatform : public Reference {

	GDCLASS(EditorExportPlatform, Reference)

public:
	typedef Error (*EditorExportSaveFunction)(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total);
	typedef Error (*EditorExportSaveSharedObject)(void *p_userdata, const SharedObject &p_so);

private:
	struct SavedData {

		uint64_t ofs;
		uint64_t size;
		Vector<uint8_t> md5;
		CharString path_utf8;

		bool operator<(const SavedData &p_data) const {
			return path_utf8 < p_data.path_utf8;
		}
	};

	struct PackData {

		FileAccess *f;
		Vector<SavedData> file_ofs;
		EditorProgress *ep;
		Vector<SharedObject> *so_files;
	};

	struct ZipData {

		void *zip;
		EditorProgress *ep;
	};

	struct FeatureContainers {
		Set<String> features;
		PoolVector<String> features_pv;
	};

	void _export_find_resources(EditorFileSystemDirectory *p_dir, Set<String> &p_paths);
	void _export_find_dependencies(const String &p_path, Set<String> &p_paths);

	void gen_debug_flags(Vector<String> &r_flags, int p_flags);
	static Error _save_pack_file(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total);
	static Error _save_zip_file(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total);

	void _edit_files_with_filter(DirAccess *da, const Vector<String> &p_filters, Set<String> &r_list, bool exclude);
	void _edit_filter_list(Set<String> &r_list, const String &p_filter, bool exclude);

	static Error _add_shared_object(void *p_userdata, const SharedObject &p_so);

protected:
	struct ExportNotifier {
		ExportNotifier(EditorExportPlatform &p_platform, const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags);
		~ExportNotifier();
	};

	FeatureContainers get_feature_containers(const Ref<EditorExportPreset> &p_preset);

	bool exists_export_template(String template_file_name, String *err) const;
	String find_export_template(String template_file_name, String *err = NULL) const;
	void gen_export_flags(Vector<String> &r_flags, int p_flags);

public:
	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) = 0;

	struct ExportOption {
		PropertyInfo option;
		Variant default_value;

		ExportOption(const PropertyInfo &p_info, const Variant &p_default) {
			option = p_info;
			default_value = p_default;
		}
		ExportOption() {}
	};

	virtual Ref<EditorExportPreset> create_preset();

	virtual void get_export_options(List<ExportOption> *r_options) = 0;
	virtual bool get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const { return true; }

	virtual String get_os_name() const = 0;
	virtual String get_name() const = 0;
	virtual Ref<Texture> get_logo() const = 0;

	Error export_project_files(const Ref<EditorExportPreset> &p_preset, EditorExportSaveFunction p_func, void *p_udata, EditorExportSaveSharedObject p_so_func = NULL);

	Error save_pack(const Ref<EditorExportPreset> &p_preset, const String &p_path, Vector<SharedObject> *p_so_files = NULL);
	Error save_zip(const Ref<EditorExportPreset> &p_preset, const String &p_path);

	virtual bool poll_devices() { return false; }
	virtual int get_device_count() const { return 0; }
	virtual String get_device_name(int p_device) const { return ""; }
	virtual String get_device_info(int p_device) const { return ""; }

	enum DebugFlags {
		DEBUG_FLAG_DUMB_CLIENT = 1,
		DEBUG_FLAG_REMOTE_DEBUG = 2,
		DEBUG_FLAG_REMOTE_DEBUG_LOCALHOST = 4,
		DEBUG_FLAG_VIEW_COLLISONS = 8,
		DEBUG_FLAG_VIEW_NAVIGATION = 16,
	};

	virtual Error run(const Ref<EditorExportPreset> &p_preset, int p_device, int p_debug_flags) { return OK; }
	virtual Ref<Texture> get_run_icon() const { return get_logo(); }

	virtual bool can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const = 0;

	virtual String get_binary_extension(const Ref<EditorExportPreset> &p_preset) const = 0;
	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0) = 0;
	virtual void get_platform_features(List<String> *r_features) = 0;

	EditorExportPlatform();
};

class EditorExportPlugin : public Reference {
	GDCLASS(EditorExportPlugin, Reference)

	friend class EditorExportPlatform;

	Vector<SharedObject> shared_objects;
	struct ExtraFile {
		String path;
		Vector<uint8_t> data;
		bool remap;
	};
	Vector<ExtraFile> extra_files;
	bool skipped;

	Vector<String> ios_frameworks;
	String ios_plist_content;
	String ios_linker_flags;
	Vector<String> ios_bundle_files;
	String ios_cpp_code;

	_FORCE_INLINE_ void _clear() {
		shared_objects.clear();
		extra_files.clear();
		skipped = false;
	}

	_FORCE_INLINE_ void _export_end() {
		ios_frameworks.clear();
		ios_bundle_files.clear();
		ios_plist_content = "";
		ios_linker_flags = "";
		ios_cpp_code = "";
	}

	void _export_file_script(const String &p_path, const String &p_type, const PoolVector<String> &p_features);
	void _export_begin_script(const PoolVector<String> &p_features, bool p_debug, const String &p_path, int p_flags);

protected:
	void add_file(const String &p_path, const Vector<uint8_t> &p_file, bool p_remap);
	void add_shared_object(const String &p_path, const Vector<String> &tags);

	void add_ios_framework(const String &p_path);
	void add_ios_plist_content(const String &p_plist_content);
	void add_ios_linker_flags(const String &p_flags);
	void add_ios_bundle_file(const String &p_path);
	void add_ios_cpp_code(const String &p_code);

	void skip();

	virtual void _export_file(const String &p_path, const String &p_type, const Set<String> &p_features);
	virtual void _export_begin(const Set<String> &p_features, bool p_debug, const String &p_path, int p_flags);

	static void _bind_methods();

public:
	Vector<String> get_ios_frameworks() const;
	String get_ios_plist_content() const;
	String get_ios_linker_flags() const;
	Vector<String> get_ios_bundle_files() const;
	String get_ios_cpp_code() const;

	EditorExportPlugin();
};

class EditorExport : public Node {
	GDCLASS(EditorExport, Node);

	Vector<Ref<EditorExportPlatform> > export_platforms;
	Vector<Ref<EditorExportPreset> > export_presets;
	Vector<Ref<EditorExportPlugin> > export_plugins;

	Timer *save_timer;
	bool block_save;

	static EditorExport *singleton;

	void _save();

protected:
	friend class EditorExportPreset;
	void save_presets();

	void _notification(int p_what);
	static void _bind_methods();

public:
	static EditorExport *get_singleton() { return singleton; }

	void add_export_platform(const Ref<EditorExportPlatform> &p_platform);
	int get_export_platform_count();
	Ref<EditorExportPlatform> get_export_platform(int p_idx);

	void add_export_preset(const Ref<EditorExportPreset> &p_preset, int p_at_pos = -1);
	int get_export_preset_count() const;
	Ref<EditorExportPreset> get_export_preset(int p_idx);
	void remove_export_preset(int p_idx);

	void add_export_plugin(const Ref<EditorExportPlugin> &p_plugin);
	void remove_export_plugin(const Ref<EditorExportPlugin> &p_plugin);
	Vector<Ref<EditorExportPlugin> > get_export_plugins();

	void load_config();

	bool poll_export_platforms();

	EditorExport();
	~EditorExport();
};

class EditorExportPlatformPC : public EditorExportPlatform {

	GDCLASS(EditorExportPlatformPC, EditorExportPlatform)

	Ref<ImageTexture> logo;
	String name;
	String os_name;
	Map<String, String> extensions;

	String release_file_32;
	String release_file_64;
	String debug_file_32;
	String debug_file_64;

	Set<String> extra_features;

	bool use64;
	int chmod_flags;

public:
	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features);

	virtual void get_export_options(List<ExportOption> *r_options);

	virtual String get_name() const;
	virtual String get_os_name() const;
	virtual Ref<Texture> get_logo() const;

	virtual bool can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const;
	virtual String get_binary_extension(const Ref<EditorExportPreset> &p_preset) const;
	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0);

	void set_extension(const String &p_extension, const String &p_feature_key = "default");
	void set_name(const String &p_name);
	void set_os_name(const String &p_name);

	void set_logo(const Ref<Texture> &p_logo);

	void set_release_64(const String &p_file);
	void set_release_32(const String &p_file);
	void set_debug_64(const String &p_file);
	void set_debug_32(const String &p_file);

	void add_platform_feature(const String &p_feature);
	virtual void get_platform_features(List<String> *r_features);

	int get_chmod_flags() const;
	void set_chmod_flags(int p_flags);

	EditorExportPlatformPC();
};

class EditorExportTextSceneToBinaryPlugin : public EditorExportPlugin {

	GDCLASS(EditorExportTextSceneToBinaryPlugin, EditorExportPlugin)

public:
	virtual void _export_file(const String &p_path, const String &p_type, const Set<String> &p_features);
	EditorExportTextSceneToBinaryPlugin();
};

#endif // EDITOR_IMPORT_EXPORT_H
