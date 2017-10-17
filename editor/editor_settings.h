/*************************************************************************/
/*  editor_settings.h                                                    */
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
#ifndef EDITOR_SETTINGS_H
#define EDITOR_SETTINGS_H

#include "object.h"

#include "core/io/config_file.h"
#include "os/thread_safe.h"
#include "resource.h"
#include "scene/gui/shortcut.h"
#include "translation.h"

class EditorPlugin;

class EditorSettings : public Resource {

	GDCLASS(EditorSettings, Resource);

private:
	_THREAD_SAFE_CLASS_

public:
	struct Plugin {

		EditorPlugin *instance;
		String path;
		String name;
		String author;
		String version;
		String description;
		bool installs;
		String script;
		Vector<String> install_files;
	};

private:
	struct VariantContainer {
		int order;
		Variant variant;
		Variant initial;
		bool hide_from_editor;
		bool save;
		VariantContainer() {
			order = 0;
			hide_from_editor = false;
			save = false;
		}
		VariantContainer(const Variant &p_variant, int p_order) {
			variant = p_variant;
			order = p_order;
			hide_from_editor = false;
		}
	};

	HashMap<String, PropertyInfo> hints;
	int last_order;
	HashMap<String, VariantContainer> props;
	String resource_path;

	bool _set(const StringName &p_name, const Variant &p_value, bool p_emit_signal = true);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	void _initial_set(const StringName &p_name, const Variant &p_value);

	static Ref<EditorSettings> singleton;

	String config_file_path;
	String settings_path;

	Ref<Resource> clipboard;

	bool save_changed_setting;

	bool optimize_save; //do not save stuff that came from config but was not set from engine

	void _load_defaults(Ref<ConfigFile> p_extra_config = NULL);
	void _load_default_text_editor_theme();

	bool _save_text_editor_theme(String p_file);

	String project_config_path;

	Vector<String> favorite_dirs;
	Vector<String> recent_dirs;

	Vector<Ref<Translation> > translations;

	Map<String, Ref<ShortCut> > shortcuts;

	void _add_property_info_bind(const Dictionary &p_info);

protected:
	static void _bind_methods();

public:
	enum {
		NOTIFICATION_EDITOR_SETTINGS_CHANGED = 10000
	};

	void set_manually(const StringName &p_name, const Variant &p_value, bool p_emit_signal = false) {
		_set(p_name, p_value, p_emit_signal);
	}

	void set_setting(const String &p_setting, const Variant &p_value);
	Variant get_setting(const String &p_setting) const;

	bool has_setting(String p_var) const;
	static EditorSettings *get_singleton();
	void erase(String p_var);
	String get_settings_path() const;
	//String get_global_settings_path() const;
	String get_project_settings_path() const;

	void setup_language();
	void setup_network();

	void raise_order(const String &p_name);
	static void create();
	static void save();
	static void destroy();

	void notify_changes();

	void set_resource_clipboard(const Ref<Resource> &p_resource) { clipboard = p_resource; }
	Ref<Resource> get_resource_clipboard() const { return clipboard; }

	void add_property_hint(const PropertyInfo &p_hint);

	void set_favorite_dirs(const Vector<String> &p_favorites_dirs);
	Vector<String> get_favorite_dirs() const;

	void set_recent_dirs(const Vector<String> &p_recent_dirs);
	Vector<String> get_recent_dirs() const;

	void load_favorites();

	void list_text_editor_themes();
	void load_text_editor_theme();
	bool import_text_editor_theme(String p_file);
	bool save_text_editor_theme();
	bool save_text_editor_theme_as(String p_file);

	Vector<String> get_script_templates(const String &p_extension);

	void add_shortcut(const String &p_name, Ref<ShortCut> &p_shortcut);
	bool is_shortcut(const String &p_name, const Ref<InputEvent> &p_event) const;
	Ref<ShortCut> get_shortcut(const String &p_name) const;
	void get_shortcut_list(List<String> *r_shortcuts);

	void set_optimize_save(bool p_optimize);

	Variant get_project_metadata(const String &p_section, const String &p_key, Variant p_default);
	void set_project_metadata(const String &p_section, const String &p_key, Variant p_data);

	bool property_can_revert(const String &p_name);
	Variant property_get_revert(const String &p_name);

	void set_initial_value(const StringName &p_name, const Variant &p_value);

	EditorSettings();
	~EditorSettings();
};

//not a macro any longer

#define EDITOR_DEF(m_var, m_val) _EDITOR_DEF(m_var, Variant(m_val))
Variant _EDITOR_DEF(const String &p_var, const Variant &p_default);

#define EDITOR_GET(m_var) _EDITOR_GET(m_var)
Variant _EDITOR_GET(const String &p_var);

#define ED_IS_SHORTCUT(p_name, p_ev) (EditorSettings::get_singleton()->is_shortcut(p_name, p_ev))
Ref<ShortCut> ED_SHORTCUT(const String &p_path, const String &p_name, uint32_t p_keycode = 0);
Ref<ShortCut> ED_GET_SHORTCUT(const String &p_path);

#endif // EDITOR_SETTINGS_H
