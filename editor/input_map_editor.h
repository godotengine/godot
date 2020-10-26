/*************************************************************************/
/*  input_map_editor.h                                                   */
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

#ifndef INPUT_MAP_EDITOR_H
#define INPUT_MAP_EDITOR_H

#include "core/undo_redo.h"
#include "editor/editor_data.h"

class InputMapEditor : public Control {
	GDCLASS(InputMapEditor, Control);

	enum InputType {
		INPUT_KEY,
		INPUT_KEY_PHYSICAL,
		INPUT_JOY_BUTTON,
		INPUT_JOY_MOTION,
		INPUT_MOUSE_BUTTON
	};

	Tree *input_editor;
	LineEdit *action_name;
	Button *action_add;
	Label *action_add_error;

	InputType add_type;
	String add_at;
	int edit_idx;

	PopupMenu *popup_add;
	ConfirmationDialog *press_a_key;
	bool press_a_key_physical;
	Label *press_a_key_label;
	ConfirmationDialog *device_input;
	OptionButton *device_id;
	OptionButton *device_index;
	Label *device_index_label;
	MenuButton *popup_copy_to_feature;

	Ref<InputEventKey> last_wait_for_key;

	AcceptDialog *message;
	UndoRedo *undo_redo;
	String inputmap_changed;
	bool setting = false;

	void _update_actions();
	void _add_item(int p_item, Ref<InputEvent> p_exiting_event = Ref<InputEvent>());
	void _edit_item(Ref<InputEvent> p_exiting_event);

	void _action_check(String p_action);
	void _action_adds(String);
	void _action_add();
	void _device_input_add();

	void _action_selected();
	void _action_edited();
	void _action_activated();
	void _action_button_pressed(Object *p_obj, int p_column, int p_id);
	void _wait_for_key(const Ref<InputEvent> &p_event);
	void _press_a_key_confirm();
	void _show_last_added(const Ref<InputEvent> &p_event, const String &p_name);

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

protected:
	int _get_current_device();
	void _set_current_device(int i_device);
	String _get_device_string(int i_device);

	void _notification(int p_what);
	static void _bind_methods();

public:
	InputMapEditor();
};

#endif // INPUT_MAP_EDITOR_H
