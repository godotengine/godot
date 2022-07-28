/*************************************************************************/
/*  editor_export_preset.h                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef EDITOR_EXPORT_PRESET_H
#define EDITOR_EXPORT_PRESET_H

class EditorExportPlatform;

#include "core/object/ref_counted.h"

class EditorExportPreset : public RefCounted {
	GDCLASS(EditorExportPreset, RefCounted);

public:
	enum ExportFilter {
		EXPORT_ALL_RESOURCES,
		EXPORT_SELECTED_SCENES,
		EXPORT_SELECTED_RESOURCES,
		EXCLUDE_SELECTED_RESOURCES,
	};

	enum ScriptExportMode {
		MODE_SCRIPT_TEXT,
		MODE_SCRIPT_COMPILED,
	};

private:
	Ref<EditorExportPlatform> platform;
	ExportFilter export_filter = EXPORT_ALL_RESOURCES;
	String include_filter;
	String exclude_filter;
	String export_path;

	String exporter;
	HashSet<String> selected_files;
	bool runnable = false;

	friend class EditorExport;
	friend class EditorExportPlatform;

	List<PropertyInfo> properties;
	HashMap<StringName, Variant> values;

	String name;

	String custom_features;

	String enc_in_filters;
	String enc_ex_filters;
	bool enc_pck = false;
	bool enc_directory = false;

	int script_mode = MODE_SCRIPT_COMPILED;
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

	void set_enc_in_filter(const String &p_filter);
	String get_enc_in_filter() const;

	void set_enc_ex_filter(const String &p_filter);
	String get_enc_ex_filter() const;

	void set_enc_pck(bool p_enabled);
	bool get_enc_pck() const;

	void set_enc_directory(bool p_enabled);
	bool get_enc_directory() const;

	void set_script_export_mode(int p_mode);
	int get_script_export_mode() const;

	void set_script_encryption_key(const String &p_key);
	String get_script_encryption_key() const;

	const List<PropertyInfo> &get_properties() const { return properties; }

	EditorExportPreset();
};

#endif // EDITOR_EXPORT_PRESET_H
