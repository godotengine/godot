/*************************************************************************/
/*  editor_settings.h                                                    */
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

#ifndef EDITOR_SETTINGS_H
#define EDITOR_SETTINGS_H

#include "core/io/config_file.h"
#include "core/io/resource.h"
#include "core/object/class_db.h"
#include "core/os/thread_safe.h"
#include "core/string/translation.h"
#include "scene/gui/shortcut.h"

class EditorPlugin;

class EditorSettings : public Resource {
	GDCLASS(EditorSettings, Resource);

	_THREAD_SAFE_CLASS_

public:
	inline static const String PROJECT_EDITOR_SETTINGS_PATH = "res://.godot/editor";
	struct Plugin {
		EditorPlugin *instance = nullptr;
		String path;
		String name;
		String author;
		String version;
		String description;
		bool installs = false;
		String script;
		Vector<String> install_files;
	};

private:
	struct VariantContainer {
		int order = 0;
		Variant variant;
		Variant initial;
		bool has_default_value = false;
		bool hide_from_editor = false;
		bool save = false;
		bool restart_if_changed = false;

		VariantContainer() {}

		VariantContainer(const Variant &p_variant, int p_order) :
				order(p_order),
				variant(p_variant) {
		}
	};

	static Ref<EditorSettings> singleton;

	HashMap<String, PropertyInfo> hints;
	HashMap<String, VariantContainer> props;
	int last_order;

	Ref<Resource> clipboard;
	mutable Map<String, Ref<Shortcut>> shortcuts;
	Map<String, List<Ref<InputEvent>>> builtin_action_overrides;

	String resource_path;
	String settings_dir;
	String data_dir;
	String cache_dir;
	String config_file_path;
	String project_config_dir;

	Vector<String> favorites;
	Vector<String> recent_dirs;

	bool save_changed_setting;
	bool optimize_save; //do not save stuff that came from config but was not set from engine

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _set_only(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _initial_set(const StringName &p_name, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _add_property_info_bind(const Dictionary &p_info);

	void _load_defaults(Ref<ConfigFile> p_extra_config = Ref<ConfigFile>());
	void _load_default_text_editor_theme();
	bool _save_text_editor_theme(String p_file);
	bool _is_default_text_editor_theme(String p_theme_name);

protected:
	static void _bind_methods();

public:
	enum {
		NOTIFICATION_EDITOR_SETTINGS_CHANGED = 10000
	};

	static EditorSettings *get_singleton();

	static void create();
	void setup_language();
	void setup_network();
	static void save();
	static void destroy();
	void set_optimize_save(bool p_optimize);

	bool has_default_value(const String &p_setting) const;
	void set_setting(const String &p_setting, const Variant &p_value);
	Variant get_setting(const String &p_setting) const;
	bool has_setting(const String &p_setting) const;
	void erase(const String &p_setting);
	void raise_order(const String &p_setting);
	void set_initial_value(const StringName &p_setting, const Variant &p_value, bool p_update_current = false);
	void set_restart_if_changed(const StringName &p_setting, bool p_restart);
	void set_manually(const StringName &p_setting, const Variant &p_value, bool p_emit_signal = false) {
		if (p_emit_signal) {
			_set(p_setting, p_value);
		} else {
			_set_only(p_setting, p_value);
		}
	}
	bool property_can_revert(const String &p_setting);
	Variant property_get_revert(const String &p_setting);
	void add_property_hint(const PropertyInfo &p_hint);

	void set_resource_clipboard(const Ref<Resource> &p_resource) { clipboard = p_resource; }
	Ref<Resource> get_resource_clipboard() const { return clipboard; }

	String get_data_dir() const;
	String get_templates_dir() const;
	String get_settings_dir() const;
	String get_project_settings_dir() const;
	String get_text_editor_themes_dir() const;
	String get_script_templates_dir() const;
	String get_project_script_templates_dir() const;
	String get_cache_dir() const;
	String get_feature_profiles_dir() const;

	void set_project_metadata(const String &p_section, const String &p_key, Variant p_data);
	Variant get_project_metadata(const String &p_section, const String &p_key, Variant p_default) const;

	void set_favorites(const Vector<String> &p_favorites);
	Vector<String> get_favorites() const;
	void set_recent_dirs(const Vector<String> &p_recent_dirs);
	Vector<String> get_recent_dirs() const;
	void load_favorites();

	bool is_dark_theme();

	void list_text_editor_themes();
	void load_text_editor_theme();
	bool import_text_editor_theme(String p_file);
	bool save_text_editor_theme();
	bool save_text_editor_theme_as(String p_file);
	bool is_default_text_editor_theme();

	Vector<String> get_script_templates(const String &p_extension, const String &p_custom_path = String());
	String get_editor_layouts_config() const;

	void add_shortcut(const String &p_name, Ref<Shortcut> &p_shortcut);
	bool is_shortcut(const String &p_name, const Ref<InputEvent> &p_event) const;
	Ref<Shortcut> get_shortcut(const String &p_name) const;
	void get_shortcut_list(List<String> *r_shortcuts);

	void set_builtin_action_override(const String &p_name, const Array &p_events);
	const Array get_builtin_action_overrides(const String &p_name) const;

	void notify_changes();

	EditorSettings();
	~EditorSettings();
};

//not a macro any longer

#define EDITOR_DEF(m_var, m_val) _EDITOR_DEF(m_var, Variant(m_val))
#define EDITOR_DEF_RST(m_var, m_val) _EDITOR_DEF(m_var, Variant(m_val), true)
Variant _EDITOR_DEF(const String &p_setting, const Variant &p_default, bool p_restart_if_changed = false);

#define EDITOR_GET(m_var) _EDITOR_GET(m_var)
Variant _EDITOR_GET(const String &p_setting);

#define ED_IS_SHORTCUT(p_name, p_ev) (EditorSettings::get_singleton()->is_shortcut(p_name, p_ev))
Ref<Shortcut> ED_SHORTCUT(const String &p_path, const String &p_name, uint32_t p_keycode = 0);
Ref<Shortcut> ED_GET_SHORTCUT(const String &p_path);

#endif // EDITOR_SETTINGS_H
