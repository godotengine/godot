/*************************************************************************/
/*  editor_import_export.h                                               */
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
#ifndef EDITOR_IMPORT_EXPORT_H
#define EDITOR_IMPORT_EXPORT_H

#include "resource.h"
#include "scene/main/node.h"
#include "scene/resources/texture.h"

class EditorExportPlatform;
class FileAccess;
struct EditorProgress;

class EditorImportPlugin : public Reference {

	OBJ_TYPE(EditorImportPlugin, Reference);

protected:
	static void _bind_methods();

	String _validate_source_path(const String &p_path);
	String _expand_source_path(const String &p_path);

public:
	static String validate_source_path(const String &p_path);
	static String expand_source_path(const String &p_path);

	virtual String get_name() const;
	virtual String get_visible_name() const;
	virtual void import_dialog(const String &p_from = "");
	virtual Error import(const String &p_path, const Ref<ResourceImportMetadata> &p_from);
	virtual void import_from_drop(const Vector<String> &p_drop, const String &p_dest_path);
	virtual void reimport_multiple_files(const Vector<String> &p_list);
	virtual bool can_reimport_multiple_files() const;
	virtual Vector<uint8_t> custom_export(const String &p_path, const Ref<EditorExportPlatform> &p_platform);

	EditorImportPlugin();
};

class EditorExportPlugin : public Reference {

	OBJ_TYPE(EditorExportPlugin, Reference);

protected:
	static void _bind_methods();

public:
	virtual Vector<uint8_t> custom_export(String &p_path, const Ref<EditorExportPlatform> &p_platform);

	EditorExportPlugin();
};

class EditorExportPlatform : public Reference {

	OBJ_TYPE(EditorExportPlatform, Reference);

public:
	typedef Error (*EditorExportSaveFunction)(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total);

private:
	bool debugging_enabled;
	int chmod_flags;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	Vector<uint8_t> get_exported_file_default(String &p_fname) const;
	virtual Vector<uint8_t> get_exported_file(String &p_fname) const;
	virtual Vector<StringName> get_dependencies(bool p_bundles) const;
	virtual String find_export_template(String template_file_name, String *err = NULL) const;
	virtual bool exists_export_template(String template_file_name, String *err = NULL) const;

	struct TempData {

		uint64_t pos;
		uint64_t ofs;
		uint64_t size;
	};

	struct PackData {

		FileAccess *ftmp;
		FileAccess *f;
		Vector<TempData> file_ofs;
		EditorProgress *ep;
		int count;
		int alignment;
	};

	struct ZipData {

		void *zip;
		EditorProgress *ep;
		int count;
	};

	void gen_export_flags(Vector<String> &r_flags, int p_flags);
	static Error save_pack_file(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total);
	static Error save_zip_file(void *p_userdata, const String &p_path, const Vector<uint8_t> &p_data, int p_file, int p_total);

public:
	enum ImageCompression {

		IMAGE_COMPRESSION_NONE,
		IMAGE_COMPRESSION_INDEXED, // used for older hardware
		IMAGE_COMPRESSION_BC, // directx compression format
		IMAGE_COMPRESSION_PVRTC, // powervr compression format
		IMAGE_COMPRESSION_PVRTC_SQUARE, // powervr compression format, square (ios)
		IMAGE_COMPRESSION_ETC1, // ericsson compression format (alpha is uncompressed)
		IMAGE_COMPRESSION_ETC2, // ericsson new compression format (can handle alpha)
	};

	enum ExportFlags {
		EXPORT_DUMB_CLIENT = 1,
		EXPORT_REMOTE_DEBUG = 2,
		EXPORT_REMOTE_DEBUG_LOCALHOST = 4,
		EXPORT_VIEW_COLLISONS = 8,
		EXPORT_VIEW_NAVIGATION = 16,
	};

	bool is_debugging_enabled() const;
	void set_debugging_enabled(bool p_enabled);

	int get_chmod_flags() const;
	void set_chmod_flags(int p_flags);

	Error export_project_files(EditorExportSaveFunction p_func, void *p_udata, bool p_make_bundles);

	Error save_pack(FileAccess *p_where, bool p_make_bundles = false, int p_alignment = 1);
	Error save_zip(const String &p_path, bool p_make_bundles = false);

	virtual String get_name() const = 0;
	virtual ImageCompression get_image_compression() const = 0;
	virtual Ref<Texture> get_logo() const = 0;

	virtual bool poll_devices() { return false; }
	virtual int get_device_count() const { return 0; }
	virtual String get_device_name(int p_device) const { return ""; }
	virtual String get_device_info(int p_device) const { return ""; }
	virtual Error run(int p_device, int p_flags) { return OK; }

	virtual bool can_export(String *r_error = NULL) const = 0;

	virtual bool requires_password(bool p_debug) const { return false; }
	virtual String get_binary_extension() const = 0;
	virtual Error export_project(const String &p_path, bool p_debug, int p_flags = 0) = 0;

	EditorExportPlatform();
};

class EditorExportPlatformPC : public EditorExportPlatform {

	OBJ_TYPE(EditorExportPlatformPC, EditorExportPlatform);

public:
	enum ExportMode {
		EXPORT_EXE,
		EXPORT_PACK,
		EXPORT_ZIP
	};

private:
	String binary_extension;
	String platform;

	String custom_release_binary;
	String custom_debug_binary;
	String release_binary32;
	String debug_binary32;
	String release_binary64;
	String debug_binary64;
	String name;
	bool use64;

	Ref<Texture> logo;

	ExportMode export_mode;
	bool bundle;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	virtual String get_name() const { return name; }
	virtual Ref<Texture> get_logo() const { return logo; }
	virtual ImageCompression get_image_compression() const { return IMAGE_COMPRESSION_BC; }

	virtual String get_binary_extension() const { return binary_extension; }
	virtual Error export_project(const String &p_path, bool p_debug, int p_flags = 0);
	virtual void set_release_binary32(const String &p_binary) { release_binary32 = p_binary; }
	virtual void set_debug_binary32(const String &p_binary) { debug_binary32 = p_binary; }
	virtual void set_release_binary64(const String &p_binary) { release_binary64 = p_binary; }
	virtual void set_debug_binary64(const String &p_binary) { debug_binary64 = p_binary; }
	virtual void set_name(const String &p_name) { name = p_name; }
	virtual void set_logo(const Ref<Texture> &p_logo) { logo = p_logo; }

	virtual bool can_export(String *r_error = NULL) const;

	void set_binary_extension(const String &p_extension);

	EditorExportPlatformPC();
};

class EditorImportExport : public Node {
	OBJ_TYPE(EditorImportExport, Node);

public:
	enum FileAction {

		ACTION_NONE,
		ACTION_COPY,
		ACTION_BUNDLE
	};

	enum ExportFilter {
		EXPORT_SELECTED,
		EXPORT_RESOURCES,
		EXPORT_ALL,

	};

	enum ImageAction {
		IMAGE_ACTION_NONE,
		IMAGE_ACTION_COMPRESS_DISK,
		IMAGE_ACTION_COMPRESS_RAM,
		IMAGE_ACTION_KEEP //for group

	};

	enum ScriptAction {
		SCRIPT_ACTION_NONE,
		SCRIPT_ACTION_COMPILE,
		SCRIPT_ACTION_ENCRYPT
	};

	enum SampleAction {

		SAMPLE_ACTION_NONE,
		SAMPLE_ACTION_COMPRESS_RAM,
	};

protected:
	struct ImageGroup {

		ImageAction action;
		bool make_atlas;
		float lossy_quality;
		float shrink;
	};

	Vector<Ref<EditorExportPlugin> > export_plugins;
	Vector<Ref<EditorImportPlugin> > plugins;
	Map<String, int> by_idx;
	ImageAction image_action;
	float image_action_compress_quality;
	float image_shrink;
	Set<String> image_formats;

	ExportFilter export_filter;
	String export_custom_filter, export_custom_filter_exclude, export_custom_filter_exclude_dir;
	Map<StringName, FileAction> files;
	Map<StringName, Ref<EditorExportPlatform> > exporters;
	Map<StringName, ImageGroup> image_groups;
	Map<StringName, StringName> image_group_files;
	Vector<String> diff_packs;

	ScriptAction script_action;
	String script_key;

	SampleAction sample_action;
	int sample_action_max_hz;
	bool sample_action_trim;

	bool convert_text_scenes;

	static EditorImportExport *singleton;

	DVector<String> _get_export_file_list();
	DVector<String> _get_export_platforms();

	static void _bind_methods();

public:
	static EditorImportExport *get_singleton() { return singleton; }

	void add_import_plugin(const Ref<EditorImportPlugin> &p_plugin);
	void remove_import_plugin(const Ref<EditorImportPlugin> &p_plugin);
	int get_import_plugin_count() const;
	Ref<EditorImportPlugin> get_import_plugin(int p_idx) const;
	Ref<EditorImportPlugin> get_import_plugin_by_name(const String &p_string) const;

	void add_export_plugin(const Ref<EditorExportPlugin> &p_plugin);
	void remove_export_plugin(const Ref<EditorExportPlugin> &p_plugin);
	int get_export_plugin_count() const;
	Ref<EditorExportPlugin> get_export_plugin(int p_idx) const;

	bool poll_export_platforms();

	void set_export_file_action(const StringName &p_export_file, FileAction p_action);
	FileAction get_export_file_action(const StringName &p_export_file) const;
	void get_export_file_list(List<StringName> *p_export_files);

	void add_export_platform(const Ref<EditorExportPlatform> &p_export);
	Ref<EditorExportPlatform> get_export_platform(const StringName &p_platform);
	void get_export_platforms(List<StringName> *r_platforms);

	void set_export_filter(ExportFilter p_enable);
	ExportFilter get_export_filter() const;

	void set_export_custom_filter(const String &p_custom_filter);
	void set_export_custom_filter_exclude(const String &p_custom_filter);
	void set_export_custom_filter_exclude_dir(const String &p_custom_filter);
	String get_export_custom_filter() const;
	String get_export_custom_filter_exclude() const;
	String get_export_custom_filter_exclude_dir() const;

	void set_export_image_action(ImageAction p_action);
	ImageAction get_export_image_action() const;

	void set_export_image_shrink(float p_shrink);
	float get_export_image_shrink() const;

	void set_export_image_quality(float p_quality);
	float get_export_image_quality() const;

	Vector<String> &get_diff_packs() { return diff_packs; }

	void image_export_group_create(const StringName &p_name);
	void image_export_group_remove(const StringName &p_name);
	bool image_export_has_group(const StringName &p_name) const;
	void image_export_get_groups(List<StringName> *r_name) const;
	void image_export_group_set_image_action(const StringName &p_export_group, ImageAction p_action);
	ImageAction image_export_group_get_image_action(const StringName &p_export_group) const;
	void image_export_group_set_make_atlas(const StringName &p_export_group, bool p_make);
	bool image_export_group_get_make_atlas(const StringName &p_export_group) const;
	void image_export_group_set_shrink(const StringName &p_export_group, float p_amount);
	float image_export_group_get_shrink(const StringName &p_export_group) const;
	void image_export_group_set_lossy_quality(const StringName &p_export_group, float p_quality);
	float image_export_group_get_lossy_quality(const StringName &p_export_group) const;

	void image_add_to_export_group(const StringName &p_image, const StringName &p_export_group);
	StringName image_get_export_group(const StringName &p_image) const;
	void image_export_get_images_in_group(const StringName &p_group, List<StringName> *r_images) const;

	Set<String> &get_image_formats() { return image_formats; }

	void script_set_action(ScriptAction p_action);
	ScriptAction script_get_action() const;

	void script_set_encryption_key(const String &p_key);
	String script_get_encryption_key() const;

	void sample_set_action(SampleAction p_action);
	SampleAction sample_get_action() const;

	void sample_set_max_hz(int p_hz);
	int sample_get_max_hz() const;

	void sample_set_trim(bool p_trim);
	bool sample_get_trim() const;

	void set_convert_text_scenes(bool p_convert);
	bool get_convert_text_scenes() const;

	void load_config();
	void save_config();

	EditorImportExport();
	~EditorImportExport();
};

VARIANT_ENUM_CAST(EditorImportExport::FileAction);
VARIANT_ENUM_CAST(EditorImportExport::ExportFilter);
VARIANT_ENUM_CAST(EditorImportExport::ImageAction);
VARIANT_ENUM_CAST(EditorImportExport::ScriptAction);
VARIANT_ENUM_CAST(EditorImportExport::SampleAction);

#endif // EDITOR_IMPORT_EXPORT_H
