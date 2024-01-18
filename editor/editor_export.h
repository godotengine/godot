/**************************************************************************/
/*  editor_export.h                                                       */
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

#ifndef EDITOR_EXPORT_H
#define EDITOR_EXPORT_H

#include "core/os/dir_access.h"
#include "core/resource.h"
#include "scene/gui/rich_text_label.h"
#include "scene/main/node.h"
#include "scene/main/timer.h"
#include "scene/resources/texture.h"

class FileAccess;
class EditorExportPlatform;
class EditorFileSystemDirectory;
struct EditorProgress;

class EditorExportPreset : public Reference {
	GDCLASS(EditorExportPreset, Reference);

public:
	enum ExportFilter {
		EXPORT_ALL_RESOURCES,
		EXPORT_SELECTED_SCENES,
		EXPORT_SELECTED_RESOURCES,
	};

	enum ScriptExportMode {
		MODE_SCRIPT_TEXT,
		MODE_SCRIPT_COMPILED,
		MODE_SCRIPT_ENCRYPTED,
	};

private:
	Ref<EditorExportPlatform> platform;
	ExportFilter export_filter;
	String include_filter;
	String exclude_filter;
	String export_path;

	String exporter;
	Set<String> selected_files;
	bool runnable;

	friend class EditorExport;
	friend class EditorExportPlatform;

	List<PropertyInfo> properties;
	Map<StringName, Variant> values;
	Map<StringName, bool> update_visibility;

	String name;

	String custom_features;

	int script_mode;
	String script_key;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	Ref<EditorExportPlatform> get_platform() const;

	bool has(const StringName &p_property) const { return values.has(p_property); }

	void update_files_to_export();

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

	void set_custom_features(const String &p_custom_features);
	String get_custom_features() const;

	void set_export_path(const String &p_path);
	String get_export_path() const;

	void set_script_export_mode(int p_mode);
	int get_script_export_mode() const;

	void set_script_encryption_key(const String &p_key);
	String get_script_encryption_key() const;

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
	GDCLASS(EditorExportPlatform, Reference);

public:
	typedef Error (*EditorExportSaveFunction)(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total);
	typedef Error (*EditorExportSaveSharedObject)(void *p_userdata, const SharedObject &p_so);

	enum ExportMessageType {
		EXPORT_MESSAGE_NONE,
		EXPORT_MESSAGE_INFO,
		EXPORT_MESSAGE_WARNING,
		EXPORT_MESSAGE_ERROR,
	};

	struct ExportMessage {
		ExportMessageType msg_type;
		String category;
		String text;
	};

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

	Vector<ExportMessage> messages;

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
	String find_export_template(String template_file_name, String *err = nullptr) const;
	void gen_export_flags(Vector<String> &r_flags, int p_flags);

public:
	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features) = 0;

	struct ExportOption {
		PropertyInfo option;
		Variant default_value;
		bool update_visibility = false;

		ExportOption(const PropertyInfo &p_info, const Variant &p_default, bool p_update_visibility = false) :
				option(p_info),
				default_value(p_default),
				update_visibility(p_update_visibility) {
		}
		ExportOption() {}
	};

	virtual Ref<EditorExportPreset> create_preset();

	virtual void clear_messages() { messages.clear(); }
	virtual void add_message(ExportMessageType p_type, const String &p_category, const String &p_message) {
		ExportMessage msg;
		msg.category = p_category;
		msg.text = p_message;
		msg.msg_type = p_type;
		messages.push_back(msg);
		switch (p_type) {
			case EXPORT_MESSAGE_INFO: {
				print_line(vformat("%s: %s", msg.category, msg.text));
			} break;
			case EXPORT_MESSAGE_WARNING: {
				WARN_PRINT(vformat("%s: %s", msg.category, msg.text));
			} break;
			case EXPORT_MESSAGE_ERROR: {
				ERR_PRINT(vformat("%s: %s", msg.category, msg.text));
			} break;
			default:
				break;
		}
	}

	virtual int get_message_count() const {
		return messages.size();
	}

	virtual ExportMessage get_message(int p_index) const {
		ERR_FAIL_INDEX_V(p_index, messages.size(), ExportMessage());
		return messages[p_index];
	}

	virtual ExportMessageType get_worst_message_type() const {
		ExportMessageType worst_type = EXPORT_MESSAGE_NONE;
		for (int i = 0; i < messages.size(); i++) {
			worst_type = MAX(worst_type, messages[i].msg_type);
		}
		return worst_type;
	}

	virtual bool fill_log_messages(RichTextLabel *p_log, Error p_err);

	virtual void get_export_options(List<ExportOption> *r_options) = 0;
	virtual bool should_update_export_options() { return false; }
	virtual bool get_option_visibility(const EditorExportPreset *p_preset, const String &p_option, const Map<StringName, Variant> &p_options) const { return true; }

	virtual String get_os_name() const = 0;
	virtual String get_name() const = 0;
	virtual Ref<Texture> get_logo() const = 0;

	Error export_project_files(const Ref<EditorExportPreset> &p_preset, EditorExportSaveFunction p_func, void *p_udata, EditorExportSaveSharedObject p_so_func = nullptr);

	Error save_pack(const Ref<EditorExportPreset> &p_preset, const String &p_path, Vector<SharedObject> *p_so_files = nullptr, bool p_embed = false, int64_t *r_embedded_start = nullptr, int64_t *r_embedded_size = nullptr);
	Error save_zip(const Ref<EditorExportPreset> &p_preset, const String &p_path);

	virtual bool poll_export() { return false; }
	virtual int get_options_count() const { return 0; }
	virtual String get_options_tooltip() const { return ""; }
	virtual Ref<ImageTexture> get_option_icon(int p_index) const;
	virtual String get_option_label(int p_device) const { return ""; }
	virtual String get_option_tooltip(int p_device) const { return ""; }

	enum DebugFlags {
		DEBUG_FLAG_DUMB_CLIENT = 1,
		DEBUG_FLAG_REMOTE_DEBUG = 2,
		DEBUG_FLAG_REMOTE_DEBUG_LOCALHOST = 4,
		DEBUG_FLAG_VIEW_COLLISONS = 8,
		DEBUG_FLAG_VIEW_NAVIGATION = 16,
		DEBUG_FLAG_SHADER_FALLBACKS = 32,
	};

	virtual Error run(const Ref<EditorExportPreset> &p_preset, int p_device, int p_debug_flags) { return OK; }
	virtual Ref<Texture> get_run_icon() const { return get_logo(); }

	String test_etc2() const; //generic test for etc2 since most platforms use it
	String test_etc2_or_pvrtc() const; // test for etc2 or pvrtc support for iOS
	bool can_export(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const;
	virtual bool has_valid_export_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const = 0;
	virtual bool has_valid_project_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error) const = 0;

	virtual List<String> get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const = 0;
	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0) = 0;
	virtual Error export_pack(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0);
	virtual Error export_zip(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0);
	virtual void get_platform_features(List<String> *r_features) = 0;
	virtual void resolve_platform_feature_priorities(const Ref<EditorExportPreset> &p_preset, Set<String> &p_features) = 0;

	EditorExportPlatform();
};

class EditorExportPlugin : public Reference {
	GDCLASS(EditorExportPlugin, Reference);

	friend class EditorExportPlatform;

	Ref<EditorExportPreset> export_preset;

	Vector<SharedObject> shared_objects;
	struct ExtraFile {
		String path;
		Vector<uint8_t> data;
		bool remap;
	};
	Vector<ExtraFile> extra_files;
	bool skipped;

	Vector<String> ios_frameworks;
	Vector<String> ios_embedded_frameworks;
	Vector<String> ios_project_static_libs;
	String ios_plist_content;
	String ios_linker_flags;
	Vector<String> ios_bundle_files;
	String ios_cpp_code;

	Vector<String> osx_plugin_files;

	_FORCE_INLINE_ void _clear() {
		shared_objects.clear();
		extra_files.clear();
		skipped = false;
	}

	_FORCE_INLINE_ void _export_end() {
		ios_frameworks.clear();
		ios_embedded_frameworks.clear();
		ios_bundle_files.clear();
		ios_plist_content = "";
		ios_linker_flags = "";
		ios_cpp_code = "";
		osx_plugin_files.clear();
	}

	void _export_file_script(const String &p_path, const String &p_type, const PoolVector<String> &p_features);
	void _export_begin_script(const PoolVector<String> &p_features, bool p_debug, const String &p_path, int p_flags);
	void _export_end_script();

protected:
	void set_export_preset(const Ref<EditorExportPreset> &p_preset);
	Ref<EditorExportPreset> get_export_preset() const;

	void add_file(const String &p_path, const Vector<uint8_t> &p_file, bool p_remap);
	void add_shared_object(const String &p_path, const Vector<String> &tags);

	void add_ios_framework(const String &p_path);
	void add_ios_embedded_framework(const String &p_path);
	void add_ios_project_static_lib(const String &p_path);
	void add_ios_plist_content(const String &p_plist_content);
	void add_ios_linker_flags(const String &p_flags);
	void add_ios_bundle_file(const String &p_path);
	void add_ios_cpp_code(const String &p_code);

	void add_osx_plugin_file(const String &p_path);

	void skip();

	virtual void _export_file(const String &p_path, const String &p_type, const Set<String> &p_features);
	virtual void _export_begin(const Set<String> &p_features, bool p_debug, const String &p_path, int p_flags);

	static void _bind_methods();

public:
	Vector<String> get_ios_frameworks() const;
	Vector<String> get_ios_embedded_frameworks() const;
	Vector<String> get_ios_project_static_libs() const;
	String get_ios_plist_content() const;
	String get_ios_linker_flags() const;
	Vector<String> get_ios_bundle_files() const;
	String get_ios_cpp_code() const;

	const Vector<String> &get_osx_plugin_files() const;

	EditorExportPlugin();
};

class EditorExport : public Node {
	GDCLASS(EditorExport, Node);

	Vector<Ref<EditorExportPlatform>> export_platforms;
	Vector<Ref<EditorExportPreset>> export_presets;
	Vector<Ref<EditorExportPlugin>> export_plugins;

	StringName _export_presets_updated;

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
	Vector<Ref<EditorExportPlugin>> get_export_plugins();

	void load_config();
	void update_export_presets();
	bool poll_export_platforms();

	EditorExport();
	~EditorExport();
};

class EditorExportPlatformPC : public EditorExportPlatform {
	GDCLASS(EditorExportPlatformPC, EditorExportPlatform);

private:
	Ref<ImageTexture> logo;
	String name;
	String os_name;
	Map<String, String> extensions;

	String release_file_32;
	String release_file_64;
	String debug_file_32;
	String debug_file_64;
	// For Linux only.
	Map<String, String> release_files;
	Map<String, String> debug_files;

	Set<String> extra_features;

	int chmod_flags;

public:
	virtual void get_preset_features(const Ref<EditorExportPreset> &p_preset, List<String> *r_features);

	virtual void get_export_options(List<ExportOption> *r_options);

	virtual String get_name() const;
	virtual String get_os_name() const;
	virtual Ref<Texture> get_logo() const;

	virtual bool has_valid_export_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error, bool &r_missing_templates) const;
	virtual bool has_valid_project_configuration(const Ref<EditorExportPreset> &p_preset, String &r_error) const;
	virtual List<String> get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const;
	virtual Error export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags = 0);
	virtual Error sign_shared_object(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path);

	virtual Error prepare_template(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags);
	virtual Error modify_template(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags) { return OK; }
	virtual Error export_project_data(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, int p_flags);
	virtual Error fixup_embedded_pck(const String &p_path, int64_t p_embedded_start, int64_t p_embedded_size) { return OK; }

	void set_extension(const String &p_extension, const String &p_feature_key = "default");
	void set_name(const String &p_name);
	void set_os_name(const String &p_name);

	void set_logo(const Ref<Texture> &p_logo);

	void set_release_64(const String &p_file);
	void set_release_32(const String &p_file);
	void set_debug_64(const String &p_file);
	void set_debug_32(const String &p_file);

	// For Linux only.
	void set_release_files(const String &p_arch, const String &p_file);
	void set_debug_files(const String &p_arch, const String &p_file);
	String get_preset_arch(const Ref<EditorExportPreset> &p_preset) const;

	void add_platform_feature(const String &p_feature);
	virtual void get_platform_features(List<String> *r_features);
	virtual void resolve_platform_feature_priorities(const Ref<EditorExportPreset> &p_preset, Set<String> &p_features);

	int get_chmod_flags() const;
	void set_chmod_flags(int p_flags);

	EditorExportPlatformPC();
};

class EditorExportTextSceneToBinaryPlugin : public EditorExportPlugin {
	GDCLASS(EditorExportTextSceneToBinaryPlugin, EditorExportPlugin);

public:
	virtual void _export_file(const String &p_path, const String &p_type, const Set<String> &p_features);
	EditorExportTextSceneToBinaryPlugin();
};

#endif // EDITOR_EXPORT_H
