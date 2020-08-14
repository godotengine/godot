/*************************************************************************/
/*  project_settings_editor.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/undo_redo.h"
#include "editor/editor_data.h"
#include "editor/editor_plugin_settings.h"
#include "editor/editor_sectioned_inspector.h"
#include "editor/input_map_editor.h"
#include "editor/localization_editor.h"
#include "editor/shader_globals_editor.h"
#include "editor_autoload_settings.h"
#include "scene/gui/tab_container.h"

class ProjectSettingsEditor : public AcceptDialog {
	GDCLASS(ProjectSettingsEditor, AcceptDialog);

	enum InputType {
		INPUT_KEY,
		INPUT_KEY_PHYSICAL,
		INPUT_JOY_BUTTON,
		INPUT_JOY_MOTION,
		INPUT_MOUSE_BUTTON
	};

	static ProjectSettingsEditor *singleton;
	ProjectSettings *ps;
	Timer *timer;

	TabContainer *tab_container;
	SectionedInspector *inspector;
	InputMapEditor *inputmap_editor;
	LocalizationEditor *localization_editor;
	EditorAutoloadSettings *autoload_settings;
	ShaderGlobalsEditor *shaders_global_variables_editor;
	EditorPluginSettings *plugin_settings;

	HBoxContainer *search_bar;
	LineEdit *search_box;
	CheckButton *advanced;

	VBoxContainer *advanced_bar;
	LineEdit *category_box;
	LineEdit *property_box;
	Button *add_button;
	Button *del_button;
	OptionButton *type;
	OptionButton *feature_override;
	Label *error_label;

	ConfirmationDialog *del_confirmation;

	Label *restart_label;
	TextureRect *restart_icon;
	PanelContainer *restart_container;
	Button *restart_close_button;

	EditorData *data;
	UndoRedo *undo_redo;

	void _advanced_pressed();
	void _update_advanced_bar();
	void _text_field_changed(const String &p_text);
	void _feature_selected(int p_index);

	String _get_setting_name() const;
	void _setting_edited(const String &p_name);
	void _setting_selected(const String &p_path);
	void _add_setting();
	void _delete_setting(bool p_confirmed);

	void _editor_restart_request();
	void _editor_restart();
	void _editor_restart_close();

	void _add_feature_overrides();
	ProjectSettingsEditor();

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
