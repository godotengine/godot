/*************************************************************************/
/*  settings_config_dialog.h                                             */
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

#ifndef SETTINGS_CONFIG_DIALOG_H
#define SETTINGS_CONFIG_DIALOG_H

#include "editor/action_map_editor.h"
#include "editor/editor_sectioned_inspector.h"
#include "editor_inspector.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/texture_rect.h"

class EditorSettingsDialog : public AcceptDialog {
	GDCLASS(EditorSettingsDialog, AcceptDialog);

	bool updating;

	TabContainer *tabs;
	Control *tab_general;
	Control *tab_shortcuts;

	LineEdit *search_box;
	LineEdit *shortcut_search_box;
	SectionedInspector *inspector;

	// Shortcuts
	enum ShortcutButton {
		SHORTCUT_ADD,
		SHORTCUT_EDIT,
		SHORTCUT_ERASE,
		SHORTCUT_REVERT
	};

	Tree *shortcuts;
	String shortcut_filter;

	InputEventConfigurationDialog *shortcut_editor;

	bool is_editing_action = false;
	String current_edited_identifier;
	Array current_events;
	int current_event_index = -1;

	Timer *timer;

	UndoRedo *undo_redo;

	virtual void cancel_pressed() override;
	virtual void ok_pressed() override;

	void _settings_changed();
	void _settings_property_edited(const String &p_name);
	void _settings_save();

	virtual void unhandled_input(const Ref<InputEvent> &p_event) override;
	void _notification(int p_what);
	void _update_icons();

	void _event_config_confirmed();

	void _create_shortcut_treeitem(TreeItem *p_parent, const String &p_shortcut_identifier, const String &p_display, Array &p_events, bool p_allow_revert, bool p_is_common, bool p_is_collapsed);
	Array _event_list_to_array_helper(List<Ref<InputEvent>> &p_events);
	void _update_builtin_action(const String &p_name, const Array &p_events);
	void _update_shortcut_events(const String &p_path, const Array &p_events);

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	void _tabs_tab_changed(int p_tab);
	void _focus_current_search_box();

	void _filter_shortcuts(const String &p_filter);

	void _update_shortcuts();
	void _shortcut_button_pressed(Object *p_item, int p_column, int p_idx);

	void _builtin_action_popup_index_pressed(int p_index);

	static void _undo_redo_callback(void *p_self, const String &p_name);

	Label *restart_label;
	TextureRect *restart_icon;
	PanelContainer *restart_container;
	Button *restart_close_button;

	void _editor_restart_request();
	void _editor_restart();
	void _editor_restart_close();

protected:
	static void _bind_methods();

public:
	void popup_edit_settings();

	EditorSettingsDialog();
	~EditorSettingsDialog();
};

#endif // SETTINGS_CONFIG_DIALOG_H
