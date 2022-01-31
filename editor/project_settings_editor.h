/*************************************************************************/
/*  project_settings_editor.h                                            */
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

#ifndef PROJECT_SETTINGS_EDITOR_H
#define PROJECT_SETTINGS_EDITOR_H

#include "core/object/undo_redo.h"
#include "editor/action_map_editor.h"
#include "editor/editor_data.h"
#include "editor/editor_plugin_settings.h"
#include "editor/editor_sectioned_inspector.h"
#include "editor/import_defaults_editor.h"
#include "editor/localization_editor.h"
#include "editor/shader_globals_editor.h"
#include "editor_autoload_settings.h"
#include "scene/gui/tab_container.h"

class ProjectSettingsEditor : public AcceptDialog {
	GDCLASS(ProjectSettingsEditor, AcceptDialog);

	static ProjectSettingsEditor *singleton;
	ProjectSettings *ps;
	Timer *timer;

	TabContainer *tab_container;
	SectionedInspector *inspector;
	ActionMapEditor *action_map;
	LocalizationEditor *localization_editor;
	EditorAutoloadSettings *autoload_settings;
	ShaderGlobalsEditor *shaders_global_variables_editor;
	EditorPluginSettings *plugin_settings;

	LineEdit *search_box;
	CheckButton *advanced;

	LineEdit *property_box;
	OptionButton *feature_box;
	OptionButton *type_box;
	Button *add_button;
	Button *del_button;

	Label *restart_label;
	TextureRect *restart_icon;
	PanelContainer *restart_container;
	Button *restart_close_button;

	ImportDefaultsEditor *import_defaults_editor;
	EditorData *data;
	UndoRedo *undo_redo;

	void _advanced_toggled(bool p_button_pressed);
	void _property_box_changed(const String &p_text);
	void _update_property_box();
	void _feature_selected(int p_index);
	void _select_type(Variant::Type p_type);

	String _get_setting_name() const;
	void _setting_edited(const String &p_name);
	void _setting_selected(const String &p_path);
	void _add_setting();
	void _delete_setting();

	void _editor_restart_request();
	void _editor_restart();
	void _editor_restart_close();

	void _add_feature_overrides();

	void _action_added(const String &p_name);
	void _action_edited(const String &p_name, const Dictionary &p_action);
	void _action_removed(const String &p_name);
	void _action_renamed(const String &p_old_name, const String &p_new_name);
	void _action_reordered(const String &p_action_name, const String &p_relative_to, bool p_before);
	void _update_action_map_editor();
	void _update_theme();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static ProjectSettingsEditor *get_singleton() { return singleton; }
	void popup_project_settings();
	void set_plugins_page();
	void update_plugins();

	EditorAutoloadSettings *get_autoload_settings() { return autoload_settings; }
	TabContainer *get_tabs() { return tab_container; }

	void queue_save();

	ProjectSettingsEditor(EditorData *p_data);
};

#endif // PROJECT_SETTINGS_EDITOR_H
