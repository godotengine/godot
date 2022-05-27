/*************************************************************************/
/*  action_map_editor.h                                                  */
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

#ifndef ACTION_MAP_EDITOR_H
#define ACTION_MAP_EDITOR_H

#include "scene/gui/check_box.h"
#include "scene/gui/check_button.h"
#include "scene/gui/color_rect.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/label.h"
#include "scene/gui/option_button.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/tree.h"

// Confirmation Dialog used when configuring an input event.
// Separate from ActionMapEditor for code cleanliness and separation of responsibilities.
class InputEventConfigurationDialog : public ConfirmationDialog {
	GDCLASS(InputEventConfigurationDialog, ConfirmationDialog);

public:
	enum InputType {
		INPUT_KEY = 1,
		INPUT_MOUSE_BUTTON = 2,
		INPUT_JOY_BUTTON = 4,
		INPUT_JOY_MOTION = 8
	};

private:
	struct IconCache {
		Ref<Texture2D> keyboard;
		Ref<Texture2D> mouse;
		Ref<Texture2D> joypad_button;
		Ref<Texture2D> joypad_axis;
	} icon_cache;

	Ref<InputEvent> event = Ref<InputEvent>();

	TabContainer *tab_container = nullptr;

	// Listening for input
	Label *event_as_text = nullptr;
	Panel *mouse_detection_rect = nullptr;

	// List of All Key/Mouse/Joypad input options.
	int allowed_input_types;
	Tree *input_list_tree = nullptr;
	LineEdit *input_list_search = nullptr;

	// Additional Options, shown depending on event selected
	VBoxContainer *additional_options_container = nullptr;

	HBoxContainer *device_container = nullptr;
	OptionButton *device_id_option = nullptr;

	HBoxContainer *mod_container = nullptr; // Contains the subcontainer and the store command checkbox.

	enum ModCheckbox {
		MOD_ALT,
		MOD_SHIFT,
		MOD_COMMAND,
		MOD_CTRL,
		MOD_META,
		MOD_MAX
	};
	String mods[MOD_MAX] = { "Alt", "Shift", "Command", "Ctrl", "Metakey" };

	CheckBox *mod_checkboxes[MOD_MAX];
	CheckBox *store_command_checkbox = nullptr;

	CheckBox *physical_key_checkbox = nullptr;

	void _set_event(const Ref<InputEvent> &p_event, bool p_update_input_list_selection = true);

	void _tab_selected(int p_tab);
	void _listen_window_input(const Ref<InputEvent> &p_event);

	void _search_term_updated(const String &p_term);
	void _update_input_list();
	void _input_list_item_selected();

	void _mod_toggled(bool p_checked, int p_index);
	void _store_command_toggled(bool p_checked);
	void _physical_keycode_toggled(bool p_checked);

	void _device_selection_changed(int p_option_button_index);
	void _set_current_device(int p_device);
	int _get_current_device() const;
	String _get_device_string(int p_device) const;

protected:
	void _notification(int p_what);

public:
	// Pass an existing event to configure it. Alternatively, pass no event to start with a blank configuration.
	void popup_and_configure(const Ref<InputEvent> &p_event = Ref<InputEvent>());
	Ref<InputEvent> get_event() const;
	String get_event_text(const Ref<InputEvent> &p_event, bool p_include_device) const;

	void set_allowed_input_types(int p_type_masks);

	InputEventConfigurationDialog();
};

class ActionMapEditor : public Control {
	GDCLASS(ActionMapEditor, Control);

public:
	struct ActionInfo {
		String name = String();
		Dictionary action = Dictionary();

		Ref<Texture2D> icon = Ref<Texture2D>();
		bool editable = true;
	};

private:
	enum ItemButton {
		BUTTON_ADD_EVENT,
		BUTTON_EDIT_EVENT,
		BUTTON_REMOVE_ACTION,
		BUTTON_REMOVE_EVENT,
	};

	Vector<ActionInfo> actions_cache;
	Tree *action_tree = nullptr;

	// Storing which action/event is currently being edited in the InputEventConfigurationDialog.

	Dictionary current_action = Dictionary();
	String current_action_name = String();
	int current_action_event_index = -1;

	// Popups

	InputEventConfigurationDialog *event_config_dialog = nullptr;
	AcceptDialog *message = nullptr;

	// Filtering and Adding actions

	bool show_builtin_actions = false;
	CheckButton *show_builtin_actions_checkbutton = nullptr;
	LineEdit *action_list_search = nullptr;

	HBoxContainer *add_hbox = nullptr;
	LineEdit *add_edit = nullptr;
	Button *add_button = nullptr;

	void _event_config_confirmed();

	void _add_action_pressed();
	void _add_edit_text_changed(const String &p_name);
	String _check_new_action_name(const String &p_name);
	bool _has_action(const String &p_name) const;
	void _add_action(const String &p_name);
	void _action_edited();

	void _tree_button_pressed(Object *p_item, int p_column, int p_id, MouseButton p_button);
	void _tree_item_activated();
	void _search_term_updated(const String &p_search_term);

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	LineEdit *get_search_box() const;
	InputEventConfigurationDialog *get_configuration_dialog();

	// Dictionary represents an Action with "events" (Array) and "deadzone" (float) items. Pass with no param to update list from cached action map.
	void update_action_list(const Vector<ActionInfo> &p_action_infos = Vector<ActionInfo>());
	void show_message(const String &p_message);

	void set_show_builtin_actions(bool p_show);

	void use_external_search_box(LineEdit *p_searchbox);

	ActionMapEditor();
};

#endif
