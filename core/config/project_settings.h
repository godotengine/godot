/*************************************************************************/
/*  project_settings.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef PROJECT_SETTINGS_H
#define PROJECT_SETTINGS_H

#include "core/object/class_db.h"
#include "core/os/thread_safe.h"
#include "core/templates/set.h"

class ProjectSettings : public Object {
	GDCLASS(ProjectSettings, Object);
	_THREAD_SAFE_CLASS_

public:
	typedef Map<String, Variant> CustomMap;
	static const String IMPORTED_FILES_PATH;

	enum {
		//properties that are not for built in values begin from this value, so builtin ones are displayed first
		NO_BUILTIN_ORDER_BASE = 1 << 16
	};

	struct AutoloadInfo {
		StringName name;
		String path;
		bool is_singleton = false;
	};

protected:
	struct VariantContainer {
		int order = 0;
		bool persist = false;
		bool basic = false;
		Variant variant;
		Variant initial;
		bool hide_from_editor = false;
		bool overridden = false;
		bool restart_if_changed = false;
#ifdef DEBUG_METHODS_ENABLED
		bool ignore_value_in_docs = false;
#endif

		VariantContainer() {}

		VariantContainer(const Variant &p_variant, int p_order, bool p_persist = false) :
				order(p_order),
				persist(p_persist),
				variant(p_variant) {
		}
	};

	int last_order = NO_BUILTIN_ORDER_BASE;
	int last_builtin_order = 0;
	uint64_t last_save_time = 0;

	Map<StringName, VariantContainer> props;
	String resource_path;
	Map<StringName, PropertyInfo> custom_prop_info;
	bool disable_feature_overrides = false;
	bool using_datapack = false;
	List<String> input_presets;

	Set<String> custom_features;
	Map<StringName, StringName> feature_overrides;

	Map<StringName, AutoloadInfo> autoloads;

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static ProjectSettings *singleton;

	Error _load_settings_text(const String &p_path);
	Error _load_settings_binary(const String &p_path);
	Error _load_settings_text_or_binary(const String &p_text_path, const String &p_bin_path);

	Error _save_settings_text(const String &p_file, const Map<String, List<String>> &props, const CustomMap &p_custom = CustomMap(), const String &p_custom_features = String());
	Error _save_settings_binary(const String &p_file, const Map<String, List<String>> &props, const CustomMap &p_custom = CustomMap(), const String &p_custom_features = String());

	Error _save_custom_bnd(const String &p_file);

	void _convert_to_last_version(int p_from_version);

	bool _load_resource_pack(const String &p_pack, bool p_replace_files = true, int p_offset = 0);

	void _add_property_info_bind(const Dictionary &p_info);

	Error _setup(const String &p_path, const String &p_main_pack, bool p_upwards = false);

	void _add_builtin_input_map();

protected:
	static void _bind_methods();

public:
	static const int CONFIG_VERSION = 4;

	void set_setting(const String &p_setting, const Variant &p_value);
	Variant get_setting(const String &p_setting) const;

	bool has_setting(String p_var) const;
	String localize_path(const String &p_path) const;
	String globalize_path(const String &p_path) const;

	void set_initial_value(const String &p_name, const Variant &p_value);
	void set_as_basic(const String &p_name, bool p_basic);
	void set_restart_if_changed(const String &p_name, bool p_restart);
	void set_ignore_value_in_docs(const String &p_name, bool p_ignore);
	bool get_ignore_value_in_docs(const String &p_name) const;

	bool property_can_revert(const String &p_name);
	Variant property_get_revert(const String &p_name);

	String get_resource_path() const;

	static ProjectSettings *get_singleton();

	void clear(const String &p_name);
	int get_order(const String &p_name) const;
	void set_order(const String &p_name, int p_order);
	void set_builtin_order(const String &p_name);
	bool is_builtin_setting(const String &p_name) const;

	Error setup(const String &p_path, const String &p_main_pack, bool p_upwards = false);

	Error save_custom(const String &p_path = "", const CustomMap &p_custom = CustomMap(), const Vector<String> &p_custom_features = Vector<String>(), bool p_merge_with_current = true);
	Error save();
	void set_custom_property_info(const String &p_prop, const PropertyInfo &p_info);
	const Map<StringName, PropertyInfo> &get_custom_property_info() const;
	uint64_t get_last_saved_time() { return last_save_time; }

	Vector<String> get_optimizer_presets() const;

	List<String> get_input_presets() const { return input_presets; }

	void set_disable_feature_overrides(bool p_disable);

	bool is_using_datapack() const;

	bool has_custom_feature(const String &p_feature) const;

	Map<StringName, AutoloadInfo> get_autoload_list() const;
	void add_autoload(const AutoloadInfo &p_autoload);
	void remove_autoload(const StringName &p_autoload);
	bool has_autoload(const StringName &p_autoload) const;
	AutoloadInfo get_autoload(const StringName &p_name) const;

	ProjectSettings();
	~ProjectSettings();
};

//not a macro any longer
Variant _GLOBAL_DEF(const String &p_var, const Variant &p_default, bool p_restart_if_changed = false, bool p_ignore_value_in_docs = false, bool p_basic = false);
#define GLOBAL_DEF(m_var, m_value) _GLOBAL_DEF(m_var, m_value)
#define GLOBAL_DEF_RST(m_var, m_value) _GLOBAL_DEF(m_var, m_value, true)
#define GLOBAL_DEF_NOVAL(m_var, m_value) _GLOBAL_DEF(m_var, m_value, false, true)
#define GLOBAL_DEF_RST_NOVAL(m_var, m_value) _GLOBAL_DEF(m_var, m_value, true, true)
#define GLOBAL_GET(m_var) ProjectSettings::get_singleton()->get(m_var)

#define GLOBAL_DEF_BASIC(m_var, m_value) _GLOBAL_DEF(m_var, m_value, false, false, true)
#define GLOBAL_DEF_RST_BASIC(m_var, m_value) _GLOBAL_DEF(m_var, m_value, true, false, true)
#define GLOBAL_DEF_NOVAL_BASIC(m_var, m_value) _GLOBAL_DEF(m_var, m_value, false, true, true)
#define GLOBAL_DEF_RST_NOVAL_BASIC(m_var, m_value) _GLOBAL_DEF(m_var, m_value, true, true, true)

#endif // PROJECT_SETTINGS_H
