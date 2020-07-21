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
#include "editor_autoload_settings.h"
#include "input_map_editor.h"
#include "localization_editor.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/tab_container.h"
#include "shader_globals_editor.h"

class ProjectSettingsEditor : public AcceptDialog {
	GDCLASS(ProjectSettingsEditor, AcceptDialog);

	enum InputType {
		INPUT_KEY,
		INPUT_KEY_PHYSICAL,
		INPUT_JOY_BUTTON,
		INPUT_JOY_MOTION,
		INPUT_MOUSE_BUTTON
	};

	TabContainer *tab_container;
	AcceptDialog *message;
	Timer *timer;

	HBoxContainer *search_bar;
	Button *search_button;
	LineEdit *search_box;
	HBoxContainer *add_prop_bar;
	LineEdit *category;
	LineEdit *property;
	OptionButton *type;

	SectionedInspector *globals_editor;

	MenuButton *popup_copy_to_feature;

	InputMapEditor *inputmap_editor;
	LocalizationEditor *localization_editor;
	EditorAutoloadSettings *autoload_settings;
	ShaderGlobalsEditor *shaders_global_variables_editor;
	EditorPluginSettings *plugin_settings;

	Label *restart_label;
	TextureRect *restart_icon;
	PanelContainer *restart_container;
	Button *restart_close_button;

	EditorData *data;
	UndoRedo *undo_redo;

	void _item_selected(const String &);
	void _item_adds(String);
	void _item_add();
	void _item_del();
	void _save();

	void _settings_prop_edited(const String &p_name);
	void _settings_changed();

	void _copy_to_platform(int p_which);
	void _copy_to_platform_about_to_show();

	void _toggle_search_bar(bool p_pressed);

	ProjectSettingsEditor();

	static ProjectSettingsEditor *singleton;

	void _editor_restart_request();
	void _editor_restart();
	void _editor_restart_close();

protected:
	void _unhandled_input(const Ref<InputEvent> &p_event);
	void _notification(int p_what);
	static void _bind_methods();

public:
	static ProjectSettingsEditor *get_singleton() { return singleton; }
	void popup_project_settings();
	void set_plugins_page();
	void update_plugins();

	EditorAutoloadSettings *get_autoload_settings() { return autoload_settings; }

	TabContainer *get_tabs();

	void queue_save();

	ProjectSettingsEditor(EditorData *p_data);
};

#endif // PROJECT_SETTINGS_EDITOR_H
